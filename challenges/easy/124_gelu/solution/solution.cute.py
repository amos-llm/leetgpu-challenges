import cutlass.cute as cute


def gelu(x):
    return 0.5 * x * (1.0 + cute.erf(x * 0.70710678118))


@cute.kernel
def gelu_kernel(X: cute.Tensor, Y: cute.Tensor, N: cute.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    idx = bidx * bdim + tidx
    if idx < N:
        Y[idx] = gelu(X[idx])


# X, Y are tensors on the GPU
@cute.jit
def solve(X: cute.Tensor, Y: cute.Tensor, N: cute.Int32):
    threads_per_block = 256
    gelu_kernel(X, Y, N).launch(
        grid=[cute.ceil_div(N, threads_per_block), 1, 1],
        block=[threads_per_block, 1, 1],
    )
