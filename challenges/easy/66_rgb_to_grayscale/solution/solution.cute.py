import cutlass.cute as cute


@cute.kernel
def rgb_to_grayscale_kernel(
    input: cute.Tensor,
    output: cute.Tensor,
    N: cute.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    idx = bidx * bdim + tidx
    if idx < N:
        r = input[idx * 3]
        g = input[idx * 3 + 1]
        b = input[idx * 3 + 2]
        output[idx] = 0.299 * r + 0.587 * g + 0.114 * b


# input, output are tensors on the GPU
@cute.jit
def solve(
    input: cute.Tensor,
    output: cute.Tensor,
    width: cute.Int32,
    height: cute.Int32,
):
    N = width * height
    threads_per_block = 256
    rgb_to_grayscale_kernel(input, output, N).launch(
        grid=[cute.ceil_div(N, threads_per_block), 1, 1],
        block=[threads_per_block, 1, 1],
    )
