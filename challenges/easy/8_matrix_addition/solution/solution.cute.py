import cutlass.cute as cute

TILE_SIZE = 16


@cute.kernel
def matrix_add_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Int32):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    col = bidx * TILE_SIZE + tidx
    row = bidy * TILE_SIZE + tidy
    if row < N and col < N:
        C[row, col] = A[row, col] + B[row, col]


# A, B, C are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Int32):
    matrix_add_kernel(A, B, C, N).launch(
        grid=[cute.ceil_div(N, TILE_SIZE), cute.ceil_div(N, TILE_SIZE), 1],
        block=[TILE_SIZE, TILE_SIZE, 1],
    )
