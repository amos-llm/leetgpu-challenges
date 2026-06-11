#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                   int num_elements) {
    int vec_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx = vec_idx * 4;
    if (global_idx + 3 < num_elements) {
        *reinterpret_cast<float4*>(&dst[global_idx]) =
            *reinterpret_cast<const float4*>(&src[global_idx]);
    } else {
        for (int i = 0; global_idx + i < num_elements; ++i) {
            dst[global_idx + i] = src[global_idx + i];
        }
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int num_elements = N * N;
    int num_vectors = (num_elements + 3) / 4;
    int threads_per_block = 256;
    int blocks_per_grid = (num_vectors + threads_per_block - 1) / threads_per_block;
    copy_matrix_kernel<<<blocks_per_grid, threads_per_block>>>(A, B, num_elements);
    cudaDeviceSynchronize();
}
