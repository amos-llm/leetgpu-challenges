import cutlass
import cutlass.cute as cute


@cute.kernel
def matrix_add_kernel(
    g_a: cute.Tensor,
    g_b: cute.Tensor,
    g_c: cute.Tensor,
    c_c: cute.Tensor,
    n: cute.Uint32,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    idx = bidx * bdim + tidx

    blk_coord = (None, bidx)
    blk_a = g_a[blk_coord]
    blk_b = g_b[blk_coord]
    blk_c = g_c[blk_coord]
    blk_crd = c_c[blk_coord]

    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), g_a.element_type, num_bits_per_copy=128
    )
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    thr_copy = tiled_copy.get_slice(tidx)
    thr_a = thr_copy.partition_S(blk_a)
    thr_b = thr_copy.partition_S(blk_b)
    thr_c = thr_copy.partition_S(blk_c)
    thr_crd = thr_copy.partition_S(blk_crd)
    vector_size = cute.size(val_layout)
    frg_a = cute.make_rmem_tensor_like(thr_a)
    frg_b = cute.make_rmem_tensor_like(thr_b)
    frg_c = cute.make_rmem_tensor_like(thr_c)
    frg_pred = cute.make_rmem_tensor((1,), cutlass.Boolean)
    frg_pred[0] = cute.elem_less(thr_crd[vector_size - 1], (n,))

    cute.copy(copy_atom, thr_a, frg_a, pred=frg_pred)
    cute.copy(copy_atom, thr_b, frg_b, pred=frg_pred)
    for i in range(cute.size(frg_c)):
        frg_c[i] = frg_a[i] + frg_b[i]
    cute.copy(copy_atom, frg_c, thr_c, pred=frg_pred)

    if idx == n // vector_size:
        for i in range(n & ~(vector_size - 1), n):
            g_c[i] = g_a[i] + g_b[i]


# A, B, C are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    n = N * N
    m_a = cute.make_tensor(A.iterator, cute.make_layout((n,), stride=(1,)))
    m_b = cute.make_tensor(B.iterator, cute.make_layout((n,), stride=(1,)))
    m_c = cute.make_tensor(C.iterator, cute.make_layout((n,), stride=(1,)))

    threads_per_block = 256
    vector_size = 128 // A.element_type.width
    thr_layout = cute.make_layout((threads_per_block,))
    val_layout = cute.make_layout((vector_size,))
    tiler, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    g_a = cute.zipped_divide(m_a, tiler)
    g_b = cute.zipped_divide(m_b, tiler)
    g_c = cute.zipped_divide(m_c, tiler)
    c_c = cute.zipped_divide(cute.make_identity_tensor(m_c.shape), tiler=tiler)

    matrix_add_kernel(g_a, g_b, g_c, c_c, n, thr_layout, val_layout).launch(
        grid=[cute.size(g_c, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )
