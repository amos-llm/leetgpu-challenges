#include <cuda_runtime.h>
#include <cute/tensor.hpp>

using namespace cute;

template <typename LayoutSrc, typename LayoutDst, typename BlockShape, typename SmemLayoutSrc,
          typename SmemLayoutDst, typename ThrLayout>
__global__ void matrix_transpose_kernel(const float* src, float* dst, LayoutSrc layout_src,
                                        LayoutDst layout_dst, BlockShape block_shape,
                                        SmemLayoutSrc smem_layout_src,
                                        SmemLayoutDst smem_layout_dst, ThrLayout thr_layout) {
    auto g_src_tensor = make_tensor(make_gmem_ptr(src), layout_src);
    auto g_dst_tensor = make_tensor(make_gmem_ptr(dst), layout_dst);

    auto src_shape = layout_src.shape();
    auto src_identity = make_identity_tensor(src_shape);
    auto g_src_pred =
        cute::lazy::transform(src_identity, [&](auto i) { return elem_less(i, src_shape); });

    auto dst_shape = layout_dst.shape();
    auto dst_identity = make_identity_tensor(dst_shape);
    auto g_dst_pred =
        cute::lazy::transform(dst_identity, [&](auto i) { return elem_less(i, dst_shape); });

    auto g_src_block = local_tile(g_src_tensor, block_shape, make_coord(blockIdx.x, blockIdx.y));
    auto g_src_pred_block = local_tile(g_src_pred, block_shape, make_coord(blockIdx.x, blockIdx.y));

    auto g_dst_block = local_tile(g_dst_tensor, block_shape, make_coord(blockIdx.y, blockIdx.x));
    auto g_dst_pred_block = local_tile(g_dst_pred, block_shape, make_coord(blockIdx.y, blockIdx.x));

    auto tg_src = local_partition(g_src_block, thr_layout, threadIdx.x);
    auto tg_src_pred = local_partition(g_src_pred_block, thr_layout, threadIdx.x);

    auto tg_dst = local_partition(g_dst_block, thr_layout, threadIdx.x);
    auto tg_dst_pred = local_partition(g_dst_pred_block, thr_layout, threadIdx.x);

    __shared__ float smem_buffer[cosize_v<SmemLayoutSrc>];

    auto s_src_tensor = make_tensor(make_smem_ptr(smem_buffer), smem_layout_src);
    auto s_dst_tensor = make_tensor(make_smem_ptr(smem_buffer), smem_layout_dst);

    auto ts_src = local_partition(s_src_tensor, thr_layout, threadIdx.x);
    auto ts_dst = local_partition(s_dst_tensor, thr_layout, threadIdx.x);

    copy_if(tg_src_pred, tg_src, ts_src);
    __syncthreads();
    copy_if(tg_dst_pred, ts_dst, tg_dst);
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    auto layout_src = make_layout(make_shape(rows, cols), GenRowMajor{});
    auto layout_dst = make_layout(make_shape(cols, rows), GenRowMajor{});

    auto block_shape = make_shape(_32{}, _32{});
    auto thr_layout = make_layout(make_shape(_8{}, _32{}), GenRowMajor{});

    auto tiled_src_layout = tiled_divide(layout_src, block_shape);
    dim3 grid_dim(size<1>(tiled_src_layout), size<2>(tiled_src_layout));
    dim3 block_dim(size(thr_layout));

    auto swizzle = Swizzle<5, 0, 5>{};
    auto smem_layout_src = composition(swizzle, make_layout(block_shape, GenRowMajor{}));
    auto smem_layout_dst = composition(swizzle, make_layout(block_shape, GenColMajor{}));

    matrix_transpose_kernel<<<grid_dim, block_dim>>>(input, output, layout_src, layout_dst,
                                                     block_shape, smem_layout_src, smem_layout_dst,
                                                     thr_layout);

    cudaDeviceSynchronize();
}
