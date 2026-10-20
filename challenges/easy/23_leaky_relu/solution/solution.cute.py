import cutlass.cute as cute


@cute.kernel
def leaky_relu_kernel(input: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    idx = bidx * bdim + tidx
    if idx < N:
        if input[idx] >= 0:
            output[idx] = input[idx]
        else:
            output[idx] = input[idx] * 0.01


# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    threads_per_block = 256
    leaky_relu_kernel(input, output, N).launch(
        grid=[cute.ceil_div(N, threads_per_block), 1, 1],
        block=[threads_per_block, 1, 1],
    )
