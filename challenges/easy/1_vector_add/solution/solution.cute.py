import cutlass
from cutlass import cute


@cute.kernel
def vector_add_kernel(
    g_a: cute.Tensor,
    g_b: cute.Tensor,
    g_c: cute.Tensor,
    crd_c: cute.Tensor,
    shape: cute.Shape,
    tiled_copy: cute.TiledCopy,
    N: cute.Uint32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    idx = bidx * bdim + tidx

    blk_coord = (None, bidx)
    blk_a = g_a[blk_coord]
    blk_b = g_b[blk_coord]
    blk_c = g_c[blk_coord]
    blk_crd = crd_c[blk_coord]

    thr_copy = tiled_copy.get_slice(tidx)
    thr_a = thr_copy.partition_S(blk_a)
    vec_size = cute.size(thr_a)
    thr_b = thr_copy.partition_S(blk_b)
    thr_c = thr_copy.partition_S(blk_c)
    thr_crd = thr_copy.partition_S(blk_crd)
    frg_a = cute.make_rmem_tensor_like(thr_a)
    frg_b = cute.make_rmem_tensor_like(thr_b)
    frg_c = cute.make_rmem_tensor_like(thr_c)
    frg_pred = cute.make_rmem_tensor((1,), cutlass.Boolean)
    frg_pred[0] = cute.elem_less(thr_crd[vec_size - 1], shape)

    cute.copy(tiled_copy, thr_a, frg_a, pred=frg_pred)
    cute.copy(tiled_copy, thr_b, frg_b, pred=frg_pred)
    for i in range(cute.size(frg_c)):
        frg_c[i] = frg_a[i] + frg_b[i]
    cute.copy(tiled_copy, frg_c, thr_c, pred=frg_pred)

    if idx == N // vec_size:
        for i in range(N & (~(vec_size - 1)), N):
            g_c[i] = g_a[i] + g_b[i]


# A, B, C are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    A = cute.make_tensor(A.iterator, cute.make_layout((N,), stride=(1,)))
    B = cute.make_tensor(B.iterator, cute.make_layout((N,), stride=(1,)))
    C = cute.make_tensor(C.iterator, cute.make_layout((N,), stride=(1,)))

    vec_size = 128 // A.element_type.width
    thr_layout = cute.make_layout((128,))
    val_layout = cute.make_layout((vec_size,))
    tiler, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        A.element_type,
        num_bits_per_copy=128,
    )
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    g_a = cute.zipped_divide(A, tiler)
    g_b = cute.zipped_divide(B, tiler)
    g_c = cute.zipped_divide(C, tiler)
    crd_c = cute.zipped_divide(cute.make_identity_tensor(C.shape), tiler)
    vector_add_kernel(g_a, g_b, g_c, crd_c, C.shape, tiled_copy, N).launch(
        grid=[cute.size(g_c, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )
