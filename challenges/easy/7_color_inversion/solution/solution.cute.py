import cutlass
import cutlass.cute as cute


@cute.kernel
def invert_kernel(
    g_image: cute.Tensor,
    c_image: cute.Tensor,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
    shape: cute.Shape,
    N: cute.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    idx = bidx * bdim + tidx

    blk_image = g_image[(None, bidx)]
    blk_crd = c_image[(None, bidx)]

    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), g_image.element_type, num_bits_per_copy=128
    )
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    thr_copy = tiled_copy.get_slice(tidx)
    thr_image = thr_copy.partition_S(blk_image)
    vector_size = cute.size(thr_image)
    thr_crd = thr_copy.partition_S(blk_crd)

    frg_image = cute.make_rmem_tensor_like(thr_image)
    frg_pred = cute.make_rmem_tensor((1,), cutlass.Boolean)
    frg_pred[0] = cute.elem_less(thr_crd[vector_size - 1], shape)

    cute.copy(copy_atom, thr_image, frg_image, pred=frg_pred)

    for px in range(0, vector_size, 4):
        frg_image[px] = (255 - frg_image[px]).to(frg_image.element_type)
        frg_image[px + 1] = (255 - frg_image[px + 1]).to(frg_image.element_type)
        frg_image[px + 2] = (255 - frg_image[px + 2]).to(frg_image.element_type)

    cute.copy(copy_atom, frg_image, thr_image, pred=frg_pred)

    if idx == N // vector_size:
        for i in range(N & (~(vector_size - 1)), N):
            if i % 4 != 3:
                g_image[i] = (255 - g_image[i]).to(g_image.element_type)


# image are tensors on the GPU
@cute.jit
def solve(image: cute.Tensor, width: cute.Int32, height: cute.Int32):
    N = width * height * 4
    m_image = cute.make_tensor(image.iterator, cute.make_layout((N,), stride=(1,)))

    dtype = m_image.element_type
    vector_size = 128 // dtype.width

    thr_layout = cute.make_layout((128,))
    val_layout = cute.make_layout((vector_size,))
    tiler, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    g_image = cute.zipped_divide(m_image, tiler)
    id_image = cute.make_identity_tensor(m_image.shape)
    c_image = cute.zipped_divide(id_image, tiler=tiler)

    invert_kernel(g_image, c_image, thr_layout, val_layout, m_image.shape, N).launch(
        grid=[cute.size(g_image, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )
