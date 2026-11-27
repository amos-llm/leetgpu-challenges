import cutlass
import cutlass.cute as cute

BM = 128
BN = 64
BK = 32
THREADS = 256
NUM_STAGES = 2


@cute.jit
def make_copy_predicates(coords: cute.Tensor, bound: cute.Int32):
    pred = cute.make_rmem_tensor(
        (
            cute.size(coords, mode=[0, 1]),
            cute.size(coords, mode=[1]),
            cute.size(coords, mode=[2]),
        ),
        cute.Boolean,
    )
    pred_residue_k = cute.make_rmem_tensor_like(pred)
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
    alpha: cute.Float32,
    beta: cute.Float32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    if cutlass.const_expr(mA.shape[1] % (128 // mA.element_type.width) == 0):
        tiled_copy_g2s_A = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mA.element_type,
                num_bits_per_copy=128,
            ),
            cute.make_layout((64, 4), stride=(4, 1)),
            cute.make_layout((1, 8)),
        )
        tiled_copy_g2s_B = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mB.element_type,
                num_bits_per_copy=128,
            ),
            cute.make_layout((8, 32), stride=(1, 8)),
            cute.make_layout((8, 1)),
        )
    else:
        tiled_copy_g2s_A = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                mA.element_type,
                num_bits_per_copy=mA.element_type.width,
            ),
            cute.make_layout((8, 32), stride=(32, 1)),
            cute.make_layout((1, 1)),
        )
        tiled_copy_g2s_B = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                mB.element_type,
                num_bits_per_copy=mB.element_type.width,
            ),
            cute.make_layout((32, 8), stride=(1, 32)),
            cute.make_layout((1, 1)),
        )
    tiled_mma = cute.make_tiled_mma(
        cute.nvgpu.warp.MmaF16BF16Op(cute.Float16, cute.Float32, (16, 8, 16)),
        cute.make_layout((2, 4, 1)),
    )

    residue_k = mA.shape[1] - cute.ceil_div(mA.shape[1], BK) * BK
    mA = cute.domain_offset((0, residue_k), mA)
    mB = cute.domain_offset((0, residue_k), mB)
    gA = cute.local_tile(mA, (BM, BK), (bidx, None))
    gB = cute.local_tile(mB, (BN, BK), (bidy, None))
    gC = cute.local_tile(mC, (BM, BN), (bidx, bidy))
    cA = cute.local_tile(cute.make_identity_tensor(mA.shape), (BM, BK), (bidx, None))
    cB = cute.local_tile(cute.make_identity_tensor(mB.shape), (BN, BK), (bidy, None))
    cC = cute.local_tile(cute.make_identity_tensor(mC.shape), (BM, BN), (bidx, bidy))

    cA = cute.domain_offset((0, residue_k, 0), cA)
    cB = cute.domain_offset((0, residue_k, 0), cB)

    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(
        mA.element_type,
        cute.make_layout((BM, BK, NUM_STAGES), stride=(BK, 1, BM * BK)),
    )
    sB = smem.allocate_tensor(
        mB.element_type,
        cute.make_layout((BN, BK, NUM_STAGES), stride=(1, BN, BN * BK)),
    )

    thr_copy_g2s_A = tiled_copy_g2s_A.get_slice(tidx)
    thr_copy_g2s_B = tiled_copy_g2s_B.get_slice(tidx)
    tAgA = thr_copy_g2s_A.partition_S(gA)
    tAcA = thr_copy_g2s_A.partition_S(cA)
    tAsA = thr_copy_g2s_A.partition_D(sA)
    tBgB = thr_copy_g2s_B.partition_S(gB)
    tBcB = thr_copy_g2s_B.partition_S(cB)
    tBsB = thr_copy_g2s_B.partition_D(sB)

    tAsA.fill(0)
    tBsB.fill(0)
    cute.arch.sync_threads()

    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC)
    tCcC = thr_mma.partition_C(cC)

    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0)

    tiled_copy_s2r_A = cute.make_tiled_copy_A(
        cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
            mA.element_type,
        ),
        tiled_mma,
    )
    tiled_copy_s2r_B = cute.make_tiled_copy_B(
        cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 2),
            mB.element_type,
        ),
        tiled_mma,
    )
    thr_copy_s2r_A = tiled_copy_s2r_A.get_slice(tidx)
    thr_copy_s2r_B = tiled_copy_s2r_B.get_slice(tidx)
    tCsA_copy_view = thr_copy_s2r_A.partition_S(sA)
    tCsB_copy_view = thr_copy_s2r_B.partition_S(sB)
    tCrA_copy_view = thr_copy_s2r_A.retile(tCrA)
    tCrB_copy_view = thr_copy_s2r_B.retile(tCrB)

    tApA, tApA_residue_k = make_copy_predicates(tAcA, mA.shape[0])
    tBpB, tBpB_residue_k = make_copy_predicates(tBcB, mB.shape[0])

    cute.copy(
        thr_copy_g2s_A,
        tAgA[None, None, None, 0],
        tAsA[None, None, None, 0],
        pred=tApA_residue_k,
    )
    cute.copy(
        thr_copy_g2s_B,
        tBgB[None, None, None, 0],
        tBsB[None, None, None, 0],
        pred=tBpB_residue_k,
    )
    cute.arch.cp_async_commit_group()

    num_k_tiles = gA.shape[2]
    for stage in cutlass.range_constexpr(1, NUM_STAGES - 1):
        if stage < num_k_tiles:
            cute.copy(
                thr_copy_g2s_A,
                tAgA[None, None, None, stage],
                tAsA[None, None, None, stage],
                pred=tApA,
            )
            cute.copy(
                thr_copy_g2s_B,
                tBgB[None, None, None, stage],
                tBsB[None, None, None, stage],
                pred=tBpB,
            )
        cute.arch.cp_async_commit_group()

    smem_pipe_read = 0
    smem_pipe_write = NUM_STAGES - 1

    for k_tile in range(num_k_tiles):
        next_k_tile = k_tile + NUM_STAGES - 1
        if next_k_tile < num_k_tiles:
            cute.copy(
                thr_copy_g2s_A,
                tAgA[None, None, None, next_k_tile],
                tAsA[None, None, None, smem_pipe_write],
                pred=tApA,
            )
            cute.copy(
                thr_copy_g2s_B,
                tBgB[None, None, None, next_k_tile],
                tBsB[None, None, None, smem_pipe_write],
                pred=tBpB,
            )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(NUM_STAGES - 1)
        cute.arch.sync_threads()

        num_k_blocks = cute.size(tCrA, mode=[2])
        cute.copy(
            tiled_copy_s2r_A,
            tCsA_copy_view[None, None, 0, smem_pipe_read],
            tCrA_copy_view[None, None, 0],
        )
        cute.copy(
            tiled_copy_s2r_B,
            tCsB_copy_view[None, None, 0, smem_pipe_read],
            tCrB_copy_view[None, None, 0],
        )
        for k_block in cutlass.range_constexpr(num_k_blocks, unroll_full=True):
            next_k_block = k_block + 1
            if cutlass.const_expr(next_k_block < num_k_blocks):
                cute.copy(
                    tiled_copy_s2r_A,
                    tCsA_copy_view[None, None, next_k_block, smem_pipe_read],
                    tCrA_copy_view[None, None, next_k_block],
                )
                cute.copy(
                    tiled_copy_s2r_B,
                    tCsB_copy_view[None, None, next_k_block, smem_pipe_read],
                    tCrB_copy_view[None, None, next_k_block],
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

    cute.arch.cp_async_wait_group(0)

    for i in range(cute.size(tCcC)):
        m, n = tCcC[i]
        if m < mC.shape[0] and n < mC.shape[1]:
            tCgC[i] = mC.element_type(alpha * tCrC[i] + beta * tCgC[i])


# A, B, C are tensors on the GPU
@cute.jit
def solve(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    M: cute.Int32,
    N: cute.Int32,
    K: cute.Int32,
    alpha: cute.Float32,
    beta: cute.Float32,
):
    B = cute.make_tensor(B.iterator, cute.select(B.layout, mode=[1, 0]))
    gemm_kernel(A, B, C, alpha, beta).launch(
        grid=[cute.ceil_div(M, BM), cute.ceil_div(N, BN), 1],
        block=[THREADS, 1, 1],
    )
