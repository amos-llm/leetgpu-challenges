import cutlass.cute as cute


def gelu(x):
    return 0.5 * x * (1 + cute.erf(x * 0.70710678118))


@cute.kernel
def geglu_kernel(input: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    idx = bidx * bdim + tidx
    if idx < N // 2:
        output[idx] = gelu(input[idx + N // 2]) * input[idx]


# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    threads_per_block = 256
    geglu_kernel(input, output, N).launch(
        grid=[cute.ceil_div(N // 2, threads_per_block), 1, 1],
        block=[threads_per_block, 1, 1],
    )
