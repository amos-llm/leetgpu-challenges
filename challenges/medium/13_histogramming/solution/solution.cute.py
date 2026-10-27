import cutlass
import cutlass.cute as cute


@cute.kernel
def reset_kernel(histogram: cute.Tensor, num_bins: cute.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bdim, _, _ = cute.arch.block_dim()

    for i in range(tidx, num_bins, bdim):
        histogram[i] = 0


@cute.kernel
def histogram_kernel(
    input: cute.Tensor,
    histogram: cute.Tensor,
    N: cute.Int32,
    num_bins: cute.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    smem_ptr = cute.arch.get_dyn_smem(cutlass.Int32)
    local_histogram = cute.make_tensor(
        smem_ptr,
        cute.make_layout((num_bins,), stride=(1,)),
    )

    for i in range(tidx, num_bins, bdim):
        local_histogram[i] = 0
    cute.arch.sync_threads()

    idx = bidx * bdim + tidx
    if idx < N:
        value = input[idx]
        if (value >= 0) and (value < num_bins):
            cute.arch.atomic_add(smem_ptr + value, 1)
    cute.arch.sync_threads()

    for i in range(tidx, num_bins, bdim):
        cute.arch.atomic_add(histogram.iterator + i, local_histogram[i])


# input, histogram are tensors on the GPU
@cute.jit
def solve(
    input: cute.Tensor,
    histogram: cute.Tensor,
    N: cute.Int32,
    num_bins: cute.Int32,
):
    threads_per_block = 256
    reset_kernel(histogram, num_bins).launch(
        grid=[1, 1, 1],
        block=[threads_per_block, 1, 1],
    )
    histogram_kernel(input, histogram, N, num_bins).launch(
        grid=[cute.ceil_div(N, threads_per_block), 1, 1],
        block=[threads_per_block, 1, 1],
        smem=num_bins * (cutlass.Int32.width // 8),
    )
