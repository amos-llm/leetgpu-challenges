import cutlass.cute as cute


@cute.kernel
def reverse_kernel(input: cute.Tensor, N: cute.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    idx = bidx * bdim + tidx
    if idx < N // 2:
        temp = input[N - idx - 1]
        input[N - idx - 1] = input[idx]
        input[idx] = temp


# input are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, N: cute.Int32):
    threads_per_block = 256
    if N > 1:
        reverse_kernel(input, N).launch(
            grid=[cute.ceil_div(N // 2, threads_per_block), 1, 1],
            block=[threads_per_block, 1, 1],
        )
