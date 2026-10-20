import cutlass
from cutlass import cute

TILE_SIZE = 32


@cute.kernel
def transpose_kernel(
    input: cute.Tensor,
    output: cute.Tensor,
    rows: cute.Int32,
    cols: cute.Int32,
):
    tx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()

    tiler = (TILE_SIZE, TILE_SIZE)
    g_i = cute.local_tile(
        input,
        tiler=tiler,
        coord=(by, bx),
    )  # (TILE_SIZE, TILE_SIZE)
    g_o = cute.local_tile(
        output,
        tiler=tiler,
        coord=(bx, by),
    )  # (TILE_SIZE, TILE_SIZE)
    c_i = cute.local_tile(
        cute.make_identity_tensor(input.shape),
        tiler=tiler,
        coord=(by, bx),
    )  # (TILE_SIZE, TILE_SIZE)
    c_o = cute.local_tile(
        cute.make_identity_tensor(output.shape),
        tiler=tiler,
        coord=(bx, by),
    )  # (TILE_SIZE, TILE_SIZE)

    smem = cutlass.utils.SmemAllocator()
    layout = cute.make_layout(
        (TILE_SIZE, TILE_SIZE),
        stride=(TILE_SIZE + 1, 1),
    )
    layout_T = cute.make_layout(
        (TILE_SIZE, TILE_SIZE),
        stride=(1, TILE_SIZE + 1),
    )
    s_i = smem.allocate_tensor(
        input.element_type,
        layout,
    )  # (TILE_SIZE, TILE_SIZE)
    s_o = cute.make_tensor(
        s_i.iterator,
        layout_T,
    )  # (TILE_SIZE, TILE_SIZE)

    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        input.element_type,
        num_bits_per_copy=input.element_type.width,
    )
    thr_layout = cute.make_layout(
        (TILE_SIZE, TILE_SIZE),
        stride=(TILE_SIZE, 1),
    )  # (TILE_SIZE, TILE_SIZE)
    val_layout = cute.make_layout((1, 1))  # (1, 1)
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    thr_copy = tiled_copy.get_slice(tx)
    t_i_g_i = thr_copy.partition_S(g_i)  # ((1, 1), 1, 1)
    t_i_s_i = thr_copy.partition_D(s_i)  # ((1, 1), 1, 1)
    t_o_s_o = thr_copy.partition_S(s_o)  # ((1, 1), 1, 1)
    t_o_g_o = thr_copy.partition_D(g_o)  # ((1, 1), 1, 1)
    t_i_c_i = thr_copy.partition_S(c_i)  # ((1, 1), 1, 1)
    t_o_c_o = thr_copy.partition_S(c_o)  # ((1, 1), 1, 1)

    t_i_p_i = cute.make_rmem_tensor(
        cute.make_layout((1, 1, 1)),
        cutlass.Boolean,
    )  # (1, 1, 1)
    t_o_p_o = cute.make_rmem_tensor(
        cute.make_layout((1, 1, 1)),
        cutlass.Boolean,
    )  # (1, 1, 1)
    t_i_p_i[0, 0, 0] = cute.elem_less(
        t_i_c_i[(0, 0), 0, 0],
        (rows, cols),
    )
    t_o_p_o[0, 0, 0] = cute.elem_less(
        t_o_c_o[(0, 0), 0, 0],
        (cols, rows),
    )

    cute.copy(tiled_copy, t_i_g_i, t_i_s_i, pred=t_i_p_i)
    cute.arch.sync_threads()
    cute.copy(tiled_copy, t_o_s_o, t_o_g_o, pred=t_o_p_o)


# input, output are tensors on the GPU
@cute.jit
def solve(
    input: cute.Tensor,
    output: cute.Tensor,
    rows: cute.Int32,
    cols: cute.Int32,
):
    transpose_kernel(input, output, rows, cols).launch(
        grid=[
            (cols + TILE_SIZE - 1) // TILE_SIZE,
            (rows + TILE_SIZE - 1) // TILE_SIZE,
            1,
        ],
        block=[TILE_SIZE * TILE_SIZE, 1, 1],
    )
