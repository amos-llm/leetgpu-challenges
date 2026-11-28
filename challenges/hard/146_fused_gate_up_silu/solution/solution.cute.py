import cutlass
import cutlass.cute as cute

BM = 128
BN = 64
BK = 32
THREADS = 256
STAGES = 2


@cute.jit
def silu(x):
    return x / (1 + cute.exp(-x))


@cute.jit
def make_copy_predicates(coords: cute.Tensor, bound: cutlass.Int32):
    pred = cute.make_rmem_tensor(
        (
            cute.size(coords, mode=[0, 1]),
            cute.size(coords, mode=[1]),
            cute.size(coords, mode=[2]),
        ),
        cutlass.Boolean,
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
def gate_up_silu_kernel(
    mX: cute.Tensor,
    mG: cute.Tensor,
    mU: cute.Tensor,
    mO: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    tiled_copy_g2s_X = cute.make_tiled_copy_tv(
        cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mX.element_type,
            num_bits_per_copy=32,
        ),
        cute.make_layout((16, 16), stride=(16, 1)),
        cute.make_layout((1, 1)),
    )
    tiled_copy_g2s_W = cute.make_tiled_copy_tv(
        cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mG.element_type,
            num_bits_per_copy=32,
        ),
        cute.make_layout((64, 4), stride=(4, 1)),
        cute.make_layout((1, 1)),
    )
    tiled_mma = cute.make_tiled_mma(
        cute.make_mma_atom(cute.nvgpu.MmaUniversalOp(cutlass.Float32)),
        cute.make_layout((16, 16, 1), stride=(16, 1, 0)),
    )

    smem = cutlass.utils.SmemAllocator()
    sX = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((BM, BK, STAGES), stride=(BK, 1, BM * BK)),
    )
    sG = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((BN, BK, STAGES), stride=(1, BN, BN * BK)),
    )
    sU = smem.allocate_tensor(cutlass.Float32, sG.layout)

    gX = cute.local_tile(mX, (BM, BK), (bidx, None))
    gG = cute.local_tile(mG, (BN, BK), (bidy, None))
    gU = cute.local_tile(mU, (BN, BK), (bidy, None))
    cX = cute.local_tile(cute.make_identity_tensor(mX.shape), (BM, BK), (bidx, None))
    cW = cute.local_tile(cute.make_identity_tensor(mG.shape), (BN, BK), (bidy, None))

    residue_k = mX.shape[1] - cute.size(gX, mode=[2]) * BK
    gX = cute.domain_offset((0, residue_k, 0), gX)
    gG = cute.domain_offset((0, residue_k, 0), gG)
    gU = cute.domain_offset((0, residue_k, 0), gU)
    cX = cute.domain_offset((0, residue_k, 0), cX)
    cW = cute.domain_offset((0, residue_k, 0), cW)

    thr_copy_X = tiled_copy_g2s_X.get_slice(tidx)
    thr_copy_W = tiled_copy_g2s_W.get_slice(tidx)
    tXgX = thr_copy_X.partition_S(gX)
    tXcX = thr_copy_X.partition_S(cX)
    tXsX = thr_copy_X.partition_D(sX)
    tGgG = thr_copy_W.partition_S(gG)
    tUgU = thr_copy_W.partition_S(gU)
    tWcW = thr_copy_W.partition_S(cW)
    tGsG = thr_copy_W.partition_D(sG)
    tUsU = thr_copy_W.partition_D(sU)

    thr_mma = tiled_mma.get_slice(tidx)
    tOsX = thr_mma.partition_A(sX)
    tOrX = tiled_mma.make_fragment_A(tOsX[None, None, None, 0])
    tOsG = thr_mma.partition_B(sG)
    tOrG = tiled_mma.make_fragment_B(tOsG[None, None, None, 0])
    tOsU = thr_mma.partition_B(sU)
    tOrU = tiled_mma.make_fragment_B(tOsU[None, None, None, 0])

    gO = cute.local_tile(mO, (BM, BN), (bidx, bidy))
    cO = cute.local_tile(cute.make_identity_tensor(mO.shape), (BM, BN), (bidx, bidy))
    tOgO = thr_mma.partition_C(gO)
    tOcO = thr_mma.partition_C(cO)
    tOrOG = thr_mma.make_fragment_C(tOgO)
    tOrOU = thr_mma.make_fragment_C(tOgO)
    tOrOG.fill(0.0)
    tOrOU.fill(0.0)

    tXpX, tXpX_residue_k = make_copy_predicates(tXcX, mX.shape[0])
    tWpW, tWpW_residue_k = make_copy_predicates(tWcW, mG.shape[0])

    num_k_tiles = cute.size(gX.shape[2])
    num_k_blocks = cute.size(tOrX, mode=[2])

    cute.copy(
        tiled_copy_g2s_X,
        tXgX[None, None, None, 0],
        tXsX[None, None, None, 0],
        pred=tXpX_residue_k,
    )
    cute.copy(
        tiled_copy_g2s_W,
        tGgG[None, None, None, 0],
        tGsG[None, None, None, 0],
        pred=tWpW_residue_k,
    )
    cute.copy(
        tiled_copy_g2s_W,
        tUgU[None, None, None, 0],
        tUsU[None, None, None, 0],
        pred=tWpW_residue_k,
    )
    cute.arch.cp_async_commit_group()

    for stage in cutlass.range_constexpr(1, STAGES - 1):
        if stage < num_k_tiles:
            cute.copy(
                tiled_copy_g2s_X,
                tXgX[None, None, None, stage],
                tXsX[None, None, None, stage],
                pred=tXpX,
            )
            cute.copy(
                tiled_copy_g2s_W,
                tGgG[None, None, None, stage],
                tGsG[None, None, None, stage],
                pred=tWpW,
            )
            cute.copy(
                tiled_copy_g2s_W,
                tUgU[None, None, None, stage],
                tUsU[None, None, None, stage],
                pred=tWpW,
            )
        cute.arch.cp_async_commit_group()

    smem_pipe_read = 0
    smem_pipe_write = STAGES - 1

    for k_tile in range(num_k_tiles):
        next_k_tile = k_tile + STAGES - 1
        if next_k_tile < num_k_tiles:
            cute.copy(
                tiled_copy_g2s_X,
                tXgX[None, None, None, next_k_tile],
                tXsX[None, None, None, smem_pipe_write],
                pred=tXpX,
            )
            cute.copy(
                tiled_copy_g2s_W,
                tGgG[None, None, None, next_k_tile],
                tGsG[None, None, None, smem_pipe_write],
                pred=tWpW,
            )
            cute.copy(
                tiled_copy_g2s_W,
                tUgU[None, None, None, next_k_tile],
                tUsU[None, None, None, smem_pipe_write],
                pred=tWpW,
            )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(STAGES - 1)
        cute.arch.sync_threads()

        cute.autovec_copy(tOsX[None, None, 0, smem_pipe_read], tOrX[None, None, 0])
        cute.autovec_copy(tOsG[None, None, 0, smem_pipe_read], tOrG[None, None, 0])
        cute.autovec_copy(tOsU[None, None, 0, smem_pipe_read], tOrU[None, None, 0])
        for k_block in cutlass.range_constexpr(num_k_blocks, unroll_full=True):
            next_k_block = k_block + 1
            if cutlass.const_expr(next_k_block < num_k_blocks):
                cute.autovec_copy(
                    tOsX[None, None, next_k_block, smem_pipe_read],
                    tOrX[None, None, next_k_block],
                )
                cute.autovec_copy(
                    tOsG[None, None, next_k_block, smem_pipe_read],
                    tOrG[None, None, next_k_block],
                )
                cute.autovec_copy(
                    tOsU[None, None, next_k_block, smem_pipe_read],
                    tOrU[None, None, next_k_block],
                )
            cute.gemm(
                tiled_mma,
                tOrOG,
                tOrX[None, None, k_block],
                tOrG[None, None, k_block],
                tOrOG,
            )
            cute.gemm(
                tiled_mma,
                tOrOU,
                tOrX[None, None, k_block],
                tOrU[None, None, k_block],
                tOrOU,
            )
        cute.arch.sync_threads()

        smem_pipe_read = (smem_pipe_read + 1) % STAGES
        smem_pipe_write = (smem_pipe_write + 1) % STAGES

    cute.arch.cp_async_wait_group(0)

    for i in cutlass.range_constexpr(cute.size(tOrOG)):
        m, n = tOcO[i]
        if m < mO.shape[0] and n < mO.shape[1]:
            tOgO[i] = silu(tOrOG[i]) * tOrOU[i]


# x, W_gate_up, output are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    W_gate_up: cute.Tensor,
    output: cute.Tensor,
    M: cute.Int32,
    H: cute.Int32,
    I: cute.Int32,
):
    gate = cute.make_tensor(
        W_gate_up.iterator,
        cute.make_layout((I, H), stride=(1, 2 * I)),
    )
    up = cute.domain_offset((I, 0), gate)
    gate_up_silu_kernel(x, gate, up, output).launch(
        grid=[cute.ceil_div(M, BM), cute.ceil_div(I, BN), 1],
        block=[THREADS, 1, 1],
    )
