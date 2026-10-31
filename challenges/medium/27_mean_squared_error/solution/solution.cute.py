import cutlass
import cutlass.cute as cute

THREADS_PER_BLOCK = 256
WARP_SIZE = 32
NUM_WARPS = THREADS_PER_BLOCK // WARP_SIZE


@cute.kernel
def reset_kernel(mse: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        mse[0] = 0.0


@cute.kernel
def mse_kernel(
    predictions: cute.Tensor,
    targets: cute.Tensor,
    mse: cute.Tensor,
    N: cute.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()

    idx = bidx * bdim + tidx
    se = 0.0
    if idx < N:
        e = predictions[idx] - targets[idx]
        se = e * e

    for offset in [16, 8, 4, 2, 1]:
        se += cute.arch.shuffle_sync_bfly(se, offset)

    allocator = cutlass.utils.SmemAllocator()
    smem = allocator.allocate_tensor(cute.Float32, cute.make_layout(NUM_WARPS))

    if lane_idx == 0:
        smem[warp_idx] = se
    cute.arch.sync_threads()

    if tidx == 0:
        sum_se = 0.0
        for i in range(NUM_WARPS):
            sum_se += smem[i]
        cute.arch.atomic_add(mse.iterator, sum_se / N)


# predictions, targets, mse are tensors on the GPU
@cute.jit
def solve(
    predictions: cute.Tensor,
    targets: cute.Tensor,
    mse: cute.Tensor,
    N: cute.Int32,
):
    reset_kernel(mse).launch(
        grid=[1, 1, 1],
        block=[1, 1, 1],
    )
    mse_kernel(predictions, targets, mse, N).launch(
        grid=[cute.ceil_div(N, THREADS_PER_BLOCK), 1, 1],
        block=[THREADS_PER_BLOCK, 1, 1],
    )
