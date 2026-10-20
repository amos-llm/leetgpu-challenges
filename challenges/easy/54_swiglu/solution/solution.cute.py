import cutlass.cute as cute


def sigmoid(x):
    return 1 / (1 + cute.exp(-x))


def silu(x):
    return x * sigmoid(x)


@cute.kernel
def swiglu_kernel(input: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    idx = bidx * bdim + tidx
    if idx < N // 2:
        output[idx] = silu(input[idx]) * input[idx + N // 2]


# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    threads_per_block = 256
    swiglu_kernel(input, output, N).launch(
        grid=[cute.ceil_div(N // 2, threads_per_block), 1, 1],
        block=[threads_per_block, 1, 1],
    )
