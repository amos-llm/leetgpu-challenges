import cutlass.cute as cute


@cute.kernel
def interleave_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    output: cute.Tensor,
    N: cute.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    idx = bidx * bdim + tidx
    if idx < N:
        output[2 * idx] = A[idx]
        output[2 * idx + 1] = B[idx]


# A, B, output are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    threads_per_block = 256
    interleave_kernel(A, B, output, N).launch(
        grid=[cute.ceil_div(N, threads_per_block), 1, 1],
        block=[threads_per_block, 1, 1],
    )
