import cutlass
import cutlass.cute as cute

TILE_SIZE = 32


@cute.kernel
def transpose_kernel(
    input: cute.Tensor,
    output: cute.Tensor,
    rows: cute.Int32,
    cols: cute.Int32,
):
    tx, ty, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()

    row = by * TILE_SIZE + ty
    col = bx * TILE_SIZE + tx

    smem = cutlass.utils.SmemAllocator()
    layout = cute.make_layout((TILE_SIZE, TILE_SIZE), stride=(TILE_SIZE + 1, 1))
    s_i = smem.allocate_tensor(input.element_type, layout)
    s_o = cute.make_tensor(
        s_i.iterator, cute.make_layout((TILE_SIZE, TILE_SIZE), stride=(1, TILE_SIZE + 1))
    )

    if row < rows and col < cols:
        s_i[ty, tx] = input[row, col]
    cute.arch.sync_threads()
    if row < rows and col < cols:
        output[col, row] = s_o[tx, ty]


# input, output are tensors on the GPU
@cute.jit
def solve(
    input: cute.Tensor,
    output: cute.Tensor,
    rows: cute.Int32,
    cols: cute.Int32,
):
    transpose_kernel(input, output, rows, cols).launch(
        grid=[cute.ceil_div(cols, TILE_SIZE), cute.ceil_div(rows, TILE_SIZE), 1],
        block=[TILE_SIZE, TILE_SIZE, 1],
    )
