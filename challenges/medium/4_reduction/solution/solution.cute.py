import cutlass
import cutlass.cute as cute

THREADS_PER_BLOCK = 256
WARP_SIZE = 32
NUM_WARPS = THREADS_PER_BLOCK // WARP_SIZE


@cute.kernel
def reset_kernel(output: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        output[0] = 0.0


@cute.kernel
def reduce_kernel(input: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    lane_id = cute.arch.lane_idx()
    warp_id = cute.arch.warp_idx()

    idx = bidx * bdim + tidx
    value = 0.0
    if idx < N:
        value = input[idx]

    for offset in [16, 8, 4, 2, 1]:
        other_value = cute.arch.shuffle_sync_bfly(value, offset)
        value += other_value

    allocator = cutlass.utils.SmemAllocator()
    smem = allocator.allocate_array(cute.Float32, NUM_WARPS)

    if lane_id == 0:
        smem[warp_id] = value
    cute.arch.sync_threads()

    if tidx == 0:
        sum = 0.0
        for w in range(NUM_WARPS):
            sum += smem[w]

        cute.arch.atomic_add(output.iterator, sum)


# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    reset_kernel(output).launch(
        grid=[1, 1, 1],
        block=[WARP_SIZE, 1, 1],
    )
    reduce_kernel(input, output, N).launch(
        grid=[cute.ceil_div(N, THREADS_PER_BLOCK), 1, 1],
        block=[THREADS_PER_BLOCK, 1, 1],
    )
