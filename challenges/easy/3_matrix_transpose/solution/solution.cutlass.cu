#include <cuda_runtime.h>
#include <cute/tensor.hpp>

using namespace cute;

template <typename LayoutS, typename LayoutD, typename CtaTiler, typename SmemLayoutS,
          typename SmemLayoutD, typename ThrLayout>
__global__ void transpose_kernel(const float* S, float* D, LayoutS layout_S, LayoutD layout_D,
                                 CtaTiler cta_tiler, SmemLayoutS smem_S_layout,
                                 SmemLayoutD smem_D_layout, ThrLayout thr_layout) {
    Tensor tensor_S = make_tensor(make_gmem_ptr(S), layout_S);
    Tensor tensor_D = make_tensor(make_gmem_ptr(D), layout_D);

    auto shape_S = shape(tensor_S);
    Tensor C_S = make_identity_tensor(shape_S);
    Tensor P_S = cute::lazy::transform(C_S, [&](auto c) { return elem_less(c, shape_S); });

    auto shape_D = shape(tensor_D);
    Tensor C_D = make_identity_tensor(shape_D);
    Tensor P_D = cute::lazy::transform(C_D, [&](auto c) { return elem_less(c, shape_D); });

    auto cta_coord_S = make_coord(blockIdx.x, blockIdx.y);
    Tensor tile_S = local_tile(tensor_S, cta_tiler, cta_coord_S);
    Tensor tile_P_S = local_tile(P_S, cta_tiler, cta_coord_S);

    auto cta_coord_D = make_coord(blockIdx.y, blockIdx.x);
    Tensor tile_D = local_tile(tensor_D, cta_tiler, cta_coord_D);
    Tensor tile_P_D = local_tile(P_D, cta_tiler, cta_coord_D);

    Tensor thr_tile_S = local_partition(tile_S, thr_layout, threadIdx.x);
    Tensor thr_tile_P_S = local_partition(tile_P_S, thr_layout, threadIdx.x);
    Tensor thr_tile_D = local_partition(tile_D, thr_layout, threadIdx.x);
    Tensor thr_tile_P_D = local_partition(tile_P_D, thr_layout, threadIdx.x);

    __shared__ float smem[cosize_v<SmemLayoutS>];

    Tensor smem_S = make_tensor(make_smem_ptr(smem), smem_S_layout);
    Tensor smem_D = make_tensor(make_smem_ptr(smem), smem_D_layout);

    Tensor thr_smem_S = local_partition(smem_S, thr_layout, threadIdx.x);
    Tensor thr_smem_D = local_partition(smem_D, thr_layout, threadIdx.x);

    copy_if(thr_tile_P_S, thr_tile_S, thr_smem_S);
    __syncthreads();
    copy_if(thr_tile_P_D, thr_smem_D, thr_tile_D);
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    auto M = int(rows);
    auto N = int(cols);

    auto layout_S = make_layout(make_shape(M, N), GenRowMajor{});
    auto layout_D = make_layout(make_shape(N, M), GenRowMajor{});

    auto bM = Int<32>{};
    auto bN = Int<32>{};
    auto cta_tiler = make_shape(bM, bN);

    auto thr_layout = make_layout(make_shape(Int<8>{}, Int<32>{}), GenRowMajor{});

    auto tiled_S = tiled_divide(layout_S, cta_tiler);
    dim3 grid_dim(size<1>(tiled_S), size<2>(tiled_S));
    dim3 block_dim(size(thr_layout));

    // 32-way bank conflict: 32 cols x 32 banks, same column in adjacent rows hits same bank
    // auto smem_S_layout = make_layout(cta_tiler, GenRowMajor{});
    // auto smem_D_layout = make_layout(cta_tiler, GenColMajor{});

    // Padding: logical shape stays (32,32), physical stride 33 breaks bank alignment
    // smem_S(r,c) = addr r*33 + c, smem_D(r,c) = addr r + c*33, both alias correctly
    // auto smem_S_layout = make_layout(cta_tiler, make_stride(Int<33>{}, Int<1>{}));
    // auto smem_D_layout = make_layout(cta_tiler, make_stride(Int<1>{}, Int<33>{}));

    // Swizzle<B=5, M=0, S=5>: XOR bit[0:4] ^ bit[5:9] -> bank = (c ^ r) % 32
    //   B=5: 32 banks = 2^5 bank select bits
    //   M=0: bank info starts at bit 0 (logical offset bit 0 = column LSB)
    //   S=5: row stride = 32 = 2^5, row info starts at bit 5
    auto swizzle = Swizzle<5, 0, 5>{};
    auto smem_S_layout = composition(swizzle, make_layout(cta_tiler, GenRowMajor{}));
    auto smem_D_layout = composition(swizzle, make_layout(cta_tiler, GenColMajor{}));

    transpose_kernel<<<grid_dim, block_dim>>>(input, output, layout_S, layout_D, cta_tiler,
                                              smem_S_layout, smem_D_layout, thr_layout);

    cudaDeviceSynchronize();
}
