import cutlass
import cutlass.cute as cute

BM = 128
BN = 128
BK = 16
THREADS = 256
WARP_SIZE = 32
WARPS = THREADS // WARP_SIZE
STAGES = 3


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
                pred_residue_k[v, mn, k] = coord[0] < bound and coord[1] >= 0
    return pred, pred_residue_k


@cute.kernel
def fused_layernorm_linear_gelu_kernel(
    mX: cute.Tensor,
    mGamma: cute.Tensor,
    mBeta: cute.Tensor,
    mW: cute.Tensor,
    mBias: cute.Tensor,
    mY: cute.Tensor,
    eps: cute.Float32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    warp = cute.arch.warp_idx()
    lane = cute.arch.lane_idx()

    M, N, K = mX.shape[0], mW.shape[0], mW.shape[1]

    smem = cutlass.utils.SmemAllocator()
    sMean = smem.allocate_tensor(
        cute.Float32,
        cute.make_layout(
            BM,
        ),
    )
    sRstd = smem.allocate_tensor(
        cute.Float32,
        cute.make_layout(
            BM,
        ),
    )

    m_tile_start = bidx * BM
    for r in range(warp, min(M - m_tile_start, BM), WARPS):
        row = m_tile_start + r
        acc = 0.0
        for col in range(lane, K, WARP_SIZE):
            acc += mX[row, col]
        for offset in (16, 8, 4, 2, 1):
            acc += cute.arch.shuffle_sync_bfly(acc, offset)
        mean = acc / K

        square_sum = 0.0
        for col in range(lane, K, WARP_SIZE):
            square_sum += (mX[row, col] - mean) * (mX[row, col] - mean)
        for offset in (16, 8, 4, 2, 1):
            square_sum += cute.arch.shuffle_sync_bfly(square_sum, offset)
        rstd = cute.math.rsqrt(square_sum / K + eps)

        if lane == 0:
            sMean[r] = mean
            sRstd[r] = rstd

    sX = smem.allocate_tensor(
        mX.element_type,
        cute.make_layout((BM, BK, STAGES), stride=(BK, 1, BM * BK)),
        byte_alignment=16,
    )
    sW = smem.allocate_tensor(
        mW.element_type,
        cute.make_layout((BN, BK, STAGES), stride=(1, BN, BN * BK)),
        byte_alignment=16,
    )

    vec_x = 4 if cutlass.const_expr(K % 4 == 0) else 1
    vec_w = 4 if cutlass.const_expr(N % 4 == 0) else 1
    tiled_copy_X = cute.make_tiled_copy_tv(
        cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mX.element_type,
            num_bits_per_copy=vec_x * mX.element_type.width,
        ),
        cute.make_layout((64, 4), stride=(4, 1)),
        cute.make_layout((1, vec_x)),
    )
    tiled_copy_W = cute.make_tiled_copy_tv(
        cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mW.element_type,
            num_bits_per_copy=vec_w * mW.element_type.width,
        ),
        cute.make_layout((16, 16), stride=(1, 16)),
        cute.make_layout((vec_w, 1)),
    )
    tiled_mma = cute.make_tiled_mma(
        cute.nvgpu.MmaUniversalOp(cute.Float32),
        cute.make_layout((16, 16, 1), stride=(16, 1, 0)),
    )

    mcX = cute.make_identity_tensor(mX.shape)
    mcW = cute.make_identity_tensor(mW.shape)
    mcY = cute.make_identity_tensor((mY.shape))

    residue_k = K - cute.ceil_div(K, BK) * BK
    mX = cute.domain_offset((0, residue_k), mX)
    mcX = cute.domain_offset((0, residue_k), mcX)
    mW = cute.domain_offset((0, residue_k), mW)
    mcW = cute.domain_offset((0, residue_k), mcW)

    gX = cute.local_tile(mX, (BM, BK), (bidx, None))
    cX = cute.local_tile(mcX, (BM, BK), (bidx, None))
    gW = cute.local_tile(mW, (BN, BK), (bidy, None))
    cW = cute.local_tile(mcW, (BN, BK), (bidy, None))
    gY = cute.local_tile(mY, (BM, BN), (bidx, bidy))
    cY = cute.local_tile(mcY, (BM, BN), (bidx, bidy))

    thr_copy_X = tiled_copy_X.get_slice(tidx)
    thr_copy_W = tiled_copy_W.get_slice(tidx)
    thr_mma = tiled_mma.get_slice(tidx)

    tXgX = thr_copy_X.partition_S(gX)
    tXcX = thr_copy_X.partition_S(cX)
    tXsX = thr_copy_X.partition_D(sX)
    tWgW = thr_copy_W.partition_S(gW)
    tWcW = thr_copy_W.partition_S(cW)
    tWsW = thr_copy_W.partition_D(sW)

    tYsX = thr_mma.partition_A(sX)
    tYrX = thr_mma.make_fragment_A(tYsX[None, None, None, 0])
    tYcX = thr_mma.partition_A(cX)
    tYsW = thr_mma.partition_B(sW)
    tYrW = thr_mma.make_fragment_B(tYsW[None, None, None, 0])
    tYgY = thr_mma.partition_C(gY)
    tYcY = thr_mma.partition_C(cY)
    tYrY = thr_mma.make_fragment_C(tYgY)
    tYrY.fill(0.0)

    tXpX, tXpX_residue_k = make_copy_predicates(tXcX, M)
    tWpW, tWpW_residue_k = make_copy_predicates(tWcW, N)

    cute.copy(
        thr_copy_X,
        tXgX[None, None, None, 0],
        tXsX[None, None, None, 0],
        pred=tXpX_residue_k,
    )
    cute.copy(
        thr_copy_W,
        tWgW[None, None, None, 0],
        tWsW[None, None, None, 0],
        pred=tWpW_residue_k,
    )
    cute.arch.cp_async_commit_group()

    num_k_tiles = gX.shape[2]
    for stage in cutlass.range_constexpr(1, STAGES - 1):
        if stage < num_k_tiles:
            cute.copy(
                thr_copy_X,
                tXgX[None, None, None, stage],
                tXsX[None, None, None, stage],
                pred=tXpX,
            )
            cute.copy(
                thr_copy_W,
                tWgW[None, None, None, stage],
                tWsW[None, None, None, stage],
                pred=tWpW,
            )
        cute.arch.cp_async_commit_group()

    smem_pipe_read = 0
    smem_pipe_write = STAGES - 1

    for k_tile in range(num_k_tiles):
        next_k_tile = k_tile + STAGES - 1
        if next_k_tile < num_k_tiles:
            cute.copy(
                thr_copy_X,
                tXgX[None, None, None, next_k_tile],
                tXsX[None, None, None, smem_pipe_write],
                pred=tXpX,
            )
            cute.copy(
                thr_copy_W,
                tWgW[None, None, None, next_k_tile],
                tWsW[None, None, None, smem_pipe_write],
                pred=tWpW,
            )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(STAGES - 1)
        cute.arch.sync_threads()

        num_k_blocks = cute.size(tYrX, mode=[2])
        cute.autovec_copy(
            tYsX[None, None, 0, smem_pipe_read],
            tYrX[None, None, 0],
        )
        cute.autovec_copy(
            tYsW[None, None, 0, smem_pipe_read],
            tYrW[None, None, 0],
        )
        for k_block in cutlass.range(num_k_blocks, unroll_full=True):
            next_k_block = k_block + 1
            if next_k_block < num_k_blocks:
                cute.autovec_copy(
                    tYsX[None, None, next_k_block, smem_pipe_read],
                    tYrX[None, None, next_k_block],
                )
                cute.autovec_copy(
                    tYsW[None, None, next_k_block, smem_pipe_read],
                    tYrW[None, None, next_k_block],
                )

            tYrX_k = tYrX[None, None, k_block]
            tYcX_k = tYcX[None, None, k_block, k_tile]
            for i in cutlass.range_constexpr(cute.size(tYrX_k)):
                row, col = tYcX_k[i]
                row -= m_tile_start
                col = cutlass.max(col, 0)
                tYrX_k[i] = (tYrX_k[i] - sMean[row]) * sRstd[row] * mGamma[col] + mBeta[col]

            cute.gemm(
                tiled_mma,
                tYrY,
                tYrX_k,
                tYrW[None, None, k_block],
                tYrY,
            )
        cute.arch.sync_threads()

        smem_pipe_read = (smem_pipe_read + 1) % STAGES
        smem_pipe_write = (smem_pipe_write + 1) % STAGES

    cute.arch.cp_async_wait_group(0)

    for i in cutlass.range_constexpr(cute.size(tYrY)):
        m, n = tYcY[i]
        if m < M and n < N:
            h = tYrY[i] + mBias[n]
            tYgY[i] = 0.5 * h * (1.0 + cute.erf(h * 0.70710678118))


# x, gamma, beta, W, bias, y are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    gamma: cute.Tensor,
    beta: cute.Tensor,
    W: cute.Tensor,
    bias: cute.Tensor,
    y: cute.Tensor,
    M: cute.Int32,
    K: cute.Int32,
    N: cute.Int32,
    eps: cute.Float32,
):
    N, K = W.shape[1], W.shape[0]
    W_t = cute.make_tensor(W.iterator, cute.make_layout((N, K), stride=(1, N)))
    fused_layernorm_linear_gelu_kernel(x, gamma, beta, W_t, bias, y, eps).launch(
        grid=[cute.ceil_div(M, BM), cute.ceil_div(N, BN), 1],
        block=[THREADS, 1, 1],
    )
