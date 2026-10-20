import cutlass.cute as cute


@cute.kernel
def clip_kernel(
    input: cute.Tensor,
    output: cute.Tensor,
    lo: cute.Float32,
    hi: cute.Float32,
    N: cute.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    idx = bidx * bdim + tidx
    if idx < N:
        output[idx] = min(hi, max(lo, input[idx]))


# input, output are tensors on the GPU
@cute.jit
def solve(
    input: cute.Tensor,
    output: cute.Tensor,
    lo: cute.Float32,
    hi: cute.Float32,
    N: cute.Int32,
):
    threads_per_block = 256
    clip_kernel(input, output, lo, hi, N).launch(
        grid=[cute.ceil_div(N, threads_per_block), 1, 1],
        block=[threads_per_block, 1, 1],
    )
