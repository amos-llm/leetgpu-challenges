import cutlass
import cutlass.cute as cute

BM = 128
BN = 128
BK = 32
THREADS = 256
NUM_STAGES = 3


@cute.jit
def make_g2s_atom(dtype: cutlass.Constexpr, vector: cutlass.Constexpr):
    return cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(
            cache_mode=(
                cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
                if vector == 4
                else cute.nvgpu.cpasync.LoadCacheMode.ALWAYS
            )
        ),
        dtype,
        num_bits_per_copy=dtype.width * vector,
    )


@cute.jit
def make_copy_predicates(coords: cute.Tensor, bound: cute.Int32):
    pred = cute.make_rmem_tensor(
        (
            cute.size(coords, mode=[0, 1]),
            cute.size(coords, mode=[1]),
            cute.size(coords, mode=[2]),
        ),
        cutlass.Boolean,
    )
    pred_residue_k = cute.make_rmem_tensor(pred.shape, cutlass.Boolean)
    for v in cutlass.range_constexpr(pred.shape[0]):
        for mn in cutlass.range_constexpr(pred.shape[1]):
            for k in cutlass.range_constexpr(pred.shape[2]):
                coord = coords[(0, v), mn, k, 0]
                pred[v, mn, k] = coord[0] < bound
                pred_residue_k[v, mn, k] = pred[v, mn, k] and coord[1] >= 0
    return pred, pred_residue_k


@cute.kernel
def gemm_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    a_vector = 1
    if cutlass.const_expr(
        mA.shape[1] % 4 == 0
        and mA.stride[1] == 1
        and mA.stride[0] % 4 == 0
        and mA.iterator.alignment >= 16
    ):
        a_vector = 4
    tiled_copy_A = cute.make_tiled_copy_tv(
        make_g2s_atom(mA.element_type, a_vector),
        cute.make_layout(
            (THREADS // BK * a_vector, BK // a_vector),
            stride=(BK // a_vector, 1),
        ),
        cute.make_layout((1, a_vector)),
    )
    b_vector = 1
    if cutlass.const_expr(
        mB.shape[0] % 4 == 0
        and mB.stride[0] == 1
        and mB.stride[1] % 4 == 0
        and mB.iterator.alignment >= 16
    ):
        b_vector = 4
    tiled_copy_B = cute.make_tiled_copy_tv(
        make_g2s_atom(mB.element_type, b_vector),
        cute.make_layout((THREADS // BK, BK), stride=(1, THREADS // BK)),
        cute.make_layout((b_vector, 1)),
    )
    tiled_mma = cute.make_tiled_mma(
        cute.make_mma_atom(cute.nvgpu.MmaUniversalOp(cute.Float32)),
        cute.make_layout((16, 16, 1), stride=(16, 1, 0)),
    )

    gA = cute.local_tile(mA, (BM, BK), (bidx, None))
    gB = cute.local_tile(mB, (BN, BK), (bidy, None))
    gC = cute.local_tile(mC, (BM, BN), (bidx, bidy))
    cA = cute.local_tile(cute.make_identity_tensor(mA.shape), (BM, BK), (bidx, None))
    cB = cute.local_tile(cute.make_identity_tensor(mB.shape), (BN, BK), (bidy, None))
    cC = cute.local_tile(cute.make_identity_tensor(mC.shape), (BM, BN), (bidx, bidy))

    residue_k = mA.shape[1] - BK * cute.size(gA, mode=[2])
    gA = cute.domain_offset((0, residue_k, 0), gA)
    gB = cute.domain_offset((0, residue_k, 0), gB)
    cA = cute.domain_offset((0, residue_k, 0), cA)
    cB = cute.domain_offset((0, residue_k, 0), cB)

    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(
        mA.element_type, cute.make_layout((BM, BK, NUM_STAGES), stride=(BK, 1, BM * BK))
    )
    sB = smem.allocate_tensor(
        mB.element_type, cute.make_layout((BN, BK, NUM_STAGES), stride=(1, BN, BN * BK))
    )

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    tAgA = thr_copy_A.partition_S(gA)
    tAcA = thr_copy_A.partition_S(cA)
    tAsA = thr_copy_A.partition_D(sA)
    tBgB = thr_copy_B.partition_S(gB)
    tBcB = thr_copy_B.partition_S(cB)
    tBsB = thr_copy_B.partition_D(sB)

    tApA, tApA_residue_k = make_copy_predicates(tAcA, mA.shape[0])
    tBpB, tBpB_residue_k = make_copy_predicates(tBcB, mB.shape[0])

    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC)
    tCcC = thr_mma.partition_C(cC)

    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgC)

    tCrC.fill(0.0)

    cute.copy(
        thr_copy_A,
        tAgA[None, None, None, 0],
        tAsA[None, None, None, 0],
        pred=tApA_residue_k,
    )
    cute.copy(
        thr_copy_B,
        tBgB[None, None, None, 0],
        tBsB[None, None, None, 0],
        pred=tBpB_residue_k,
    )
    cute.arch.cp_async_commit_group()

    smem_pipe_read = 0
    smem_pipe_write = NUM_STAGES - 1

    num_k_tiles = cute.size(gA, mode=[2])
    for stage in cutlass.range_constexpr(1, NUM_STAGES - 1):
        if stage < num_k_tiles:
            cute.copy(
                thr_copy_A,
                tAgA[None, None, None, stage],
                tAsA[None, None, None, stage],
                pred=tApA,
            )
            cute.copy(
                thr_copy_B,
                tBgB[None, None, None, stage],
                tBsB[None, None, None, stage],
                pred=tBpB,
            )
            cute.arch.cp_async_commit_group()

    for k_tile in range(num_k_tiles):
        next_k_tile = k_tile + NUM_STAGES - 1
        if next_k_tile < num_k_tiles:
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, next_k_tile],
                tAsA[None, None, None, smem_pipe_write],
                pred=tApA,
            )
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, next_k_tile],
                tBsB[None, None, None, smem_pipe_write],
                pred=tBpB,
            )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(NUM_STAGES - 1)
        else:
            cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        cute.autovec_copy(tCsA[None, None, 0, smem_pipe_read], tCrA[None, None, 0])
        cute.autovec_copy(tCsB[None, None, 0, smem_pipe_read], tCrB[None, None, 0])

        for k_block in cutlass.range_constexpr(tCrA.shape[2], unroll_full=True):
            next_k_block = k_block + 1
            if next_k_block < tCrA.shape[2]:
                cute.autovec_copy(
                    tCsA[None, None, next_k_block, smem_pipe_read],
                    tCrA[None, None, next_k_block],
                )
                cute.autovec_copy(
                    tCsB[None, None, next_k_block, smem_pipe_read],
                    tCrB[None, None, next_k_block],
                )
            cute.gemm(
                tiled_mma,
                tCrC,
                tCrA[None, None, k_block],
                tCrB[None, None, k_block],
                tCrC,
            )
        cute.arch.sync_threads()

        smem_pipe_read = (smem_pipe_read + 1) % NUM_STAGES
        smem_pipe_write = (smem_pipe_write + 1) % NUM_STAGES

    for i in range(cute.size(tCcC)):
        m, n = tCcC[i]
        if m < mC.shape[0] and n < mC.shape[1]:
            tCgC[i] = tCrC[i]


@cute.jit
def solve(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    M: cute.Int32,
    N: cute.Int32,
    K: cute.Int32,
):
    N, K = B.shape[1], B.shape[0]
    B_t = cute.make_tensor(B.iterator, cute.make_layout((N, K), stride=(1, N)))
    gemm_kernel(A, B_t, C).launch(
        grid=[cute.ceil_div(M, BM), cute.ceil_div(N, BN), 1],
        block=[THREADS, 1, 1],
    )
