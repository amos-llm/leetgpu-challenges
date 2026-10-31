import cutlass.cute as cute

TILE_SIZE = 16


@cute.kernel
def copy_kernel(A: cute.Tensor, B: cute.Tensor, N: cute.Int32):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    col = bidx * TILE_SIZE + tidx
    row = bidy * TILE_SIZE + tidy
    if row < N and col < N:
        B[row, col] = A[row, col]


# A, B are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, N: cute.Int32):
    copy_kernel(A, B, N).launch(
        grid=[cute.ceil_div(N, TILE_SIZE), cute.ceil_div(N, TILE_SIZE), 1],
        block=[TILE_SIZE, TILE_SIZE, 1],
    )
