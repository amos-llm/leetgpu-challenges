import cutlass.cute as cute


@cute.kernel
def invert_kernel(image: cute.Tensor, N: cute.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    idx = bidx * bdim + tidx
    if idx < N:
        base = idx * 4
        image[base] = cute.Uint8(255) - image[base]
        image[base + 1] = cute.Uint8(255) - image[base + 1]
        image[base + 2] = cute.Uint8(255) - image[base + 2]


# image are tensors on the GPU
@cute.jit
def solve(image: cute.Tensor, width: cute.Int32, height: cute.Int32):
    N = width * height
    threads_per_block = 256
    invert_kernel(image, N).launch(
        grid=[cute.ceil_div(N, threads_per_block), 1, 1],
        block=[threads_per_block, 1, 1],
    )
