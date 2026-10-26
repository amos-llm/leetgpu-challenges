import cutlass
import cutlass.cute as cute

THREADS_PER_BLOCK = 1024
WARP_SIZE = 32
NUM_WARPS = THREADS_PER_BLOCK // WARP_SIZE


@cute.kernel
def softmax_kernel(input: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bdim, _, _ = cute.arch.block_dim()
    lane_id = cute.arch.lane_idx()
    warp_id = cute.arch.warp_idx()

    local_max = -1e30
    for idx in range(tidx, N, bdim):
        value = input[idx]
        if value > local_max:
            local_max = value

    for offset in [16, 8, 4, 2, 1]:
        other_max = cute.arch.shuffle_sync_bfly(local_max, offset)
        if other_max > local_max:
            local_max = other_max

    allocator = cutlass.utils.SmemAllocator()
    smem = allocator.allocate_tensor(
        cute.Float32,
        cute.make_layout((NUM_WARPS,), stride=(1,)),
    )

    if lane_id == 0:
        smem[warp_id] = local_max
    cute.arch.sync_threads()

    block_max = allocator.allocate_tensor(
        cute.Float32,
        cute.make_layout((1,), stride=(1,)),
    )
    if tidx == 0:
        for i in range(NUM_WARPS):
            if smem[i] > local_max:
                local_max = smem[i]
        block_max[0] = local_max
    cute.arch.sync_threads()

    local_sum = 0.0
    for idx in range(tidx, N, THREADS_PER_BLOCK):
        local_sum += cute.math.exp(input[idx] - block_max[0])
    for offset in [16, 8, 4, 2, 1]:
        local_sum += cute.arch.shuffle_sync_bfly(local_sum, offset)

    if lane_id == 0:
        smem[warp_id] = local_sum
    cute.arch.sync_threads()

    block_sum = allocator.allocate_tensor(
        cute.Float32,
        cute.make_layout((1,), stride=(1,)),
    )
    if tidx == 0:
        sum = 0.0
        for i in range(NUM_WARPS):
            sum += smem[i]
        block_sum[0] = sum
    cute.arch.sync_threads()

    for idx in range(tidx, N, THREADS_PER_BLOCK):
        output[idx] = cute.math.exp(input[idx] - block_max[0]) / block_sum[0]


# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    softmax_kernel(input, output, N).launch(
        grid=[cute.ceil_div(N, THREADS_PER_BLOCK), 1, 1],
        block=[THREADS_PER_BLOCK, 1, 1],
    )
