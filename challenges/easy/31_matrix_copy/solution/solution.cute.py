import cutlass
import cutlass.cute as cute


@cute.kernel
def copy_kernel(
    g_a: cute.Tensor,
    g_b: cute.Tensor,
    c_b: cute.Tensor,
    n: cute.Int32,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    blk_coord = (None, bidx)
    blk_a = g_a[blk_coord]
    blk_b = g_b[blk_coord]
    blk_c = c_b[blk_coord]

    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), g_a.element_type, num_bits_per_copy=128
    )
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    thr_copy = tiled_copy.get_slice(tidx)
    t_a = thr_copy.partition_S(blk_a)
    t_b = thr_copy.partition_D(blk_b)
    t_c = thr_copy.partition_S(blk_c)

    vector_size = cute.size(val_layout)
    frg_pred = cute.make_rmem_tensor((1,), cutlass.Boolean)
    frg_pred[0] = cute.elem_less(t_c[vector_size - 1], (n,))

    cute.copy(copy_atom, t_a, t_b, pred=frg_pred)

    idx = bidx * bdim + tidx
    if idx == n // vector_size:
        for i in range(n & ~(vector_size - 1), n):
            g_b[i] = g_a[i]


# A, B are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, N: cute.Int32):
    n = N * N
    m_a = cute.make_tensor(A.iterator, cute.make_layout((n,), stride=(1,)))
    m_b = cute.make_tensor(B.iterator, cute.make_layout((n,), stride=(1,)))

    threads_per_block = 128
    vector_size = 4
    thr_layout = cute.make_layout((threads_per_block,))
    val_layout = cute.make_layout((vector_size,))
    tiler, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    g_a = cute.zipped_divide(m_a, tiler)
    g_b = cute.zipped_divide(m_b, tiler)
    c_b = cute.zipped_divide(cute.make_identity_tensor(m_b.shape), tiler=tiler)

    copy_kernel(g_a, g_b, c_b, n, thr_layout, val_layout).launch(
        grid=[cute.size(g_b, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )
