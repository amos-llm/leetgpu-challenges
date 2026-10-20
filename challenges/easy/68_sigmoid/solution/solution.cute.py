import cutlass.cute as cute


def sigmoid(x):
    return 1 / (1 + cute.exp(-x))


@cute.kernel
def sigmoid_kernel(X: cute.Tensor, Y: cute.Tensor, N: cute.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    idx = bidx * bdim + tidx
    if idx < N:
        Y[idx] = sigmoid(X[idx])


# X, Y are tensors on the GPU
@cute.jit
def solve(X: cute.Tensor, Y: cute.Tensor, N: cute.Int32):
    threads_per_block = 256
    sigmoid_kernel(X, Y, N).launch(
        grid=[cute.ceil_div(N, threads_per_block), 1, 1],
        block=[threads_per_block, 1, 1],
    )
