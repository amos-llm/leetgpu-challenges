import cutlass
import cutlass.cute as cute

THREADS_PER_BLOCK = 256
WARP_SIZE = 32
NUM_WARPS = THREADS_PER_BLOCK // WARP_SIZE


@cute.kernel
def rms_norm_kernel(
    x: cute.Tensor,
    residual: cute.Tensor,
    weight: cute.Tensor,
    out: cute.Tensor,
    C: cute.Int32,
    eps: cute.Float32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()

    square_sum = 0.0
    for c in range(tidx, C, bdim):
        z = x[bidx, c] + residual[bidx, c]
        square_sum += z * z

    for offset in [16, 8, 4, 2, 1]:
        square_sum += cute.arch.shuffle_sync_bfly(square_sum, offset)
    smem = cutlass.utils.SmemAllocator().allocate_tensor(
        cute.Float32, cute.make_layout((NUM_WARPS,))
    )
    if lane_idx == 0:
        smem[warp_idx] = square_sum
    cute.arch.sync_threads()

    square_sum = 0.0
    for i in range(NUM_WARPS):
        square_sum += smem[i]
    rms = cute.math.rsqrt(square_sum / C + eps)

    for c in range(tidx, C, bdim):
        z = x[bidx, c] + residual[bidx, c]
        out[bidx, c] = z * rms * weight[c]


# x, residual, weight, out are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    residual: cute.Tensor,
    weight: cute.Tensor,
    out: cute.Tensor,
    N: cute.Int32,
    C: cute.Int32,
    eps: cute.Float32,
):
    rms_norm_kernel(x, residual, weight, out, C, eps).launch(
        grid=[N, 1, 1],
        block=[THREADS_PER_BLOCK, 1, 1],
    )
