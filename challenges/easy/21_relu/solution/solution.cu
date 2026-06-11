#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int vec_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx = vec_idx * 4;
    if (global_idx + 3 < N) {
        float4 val = *reinterpret_cast<const float4*>(&input[global_idx]);
        val.x = fmaxf(0.0f, val.x);
        val.y = fmaxf(0.0f, val.y);
        val.z = fmaxf(0.0f, val.z);
        val.w = fmaxf(0.0f, val.w);
        *reinterpret_cast<float4*>(&output[global_idx]) = val;
    } else {
        for (int i = 0; global_idx + i < N; ++i) {
            output[global_idx + i] = fmaxf(0.0f, input[global_idx + i]);
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int num_vectors = (N + 3) / 4;
    int threads_per_block = 256;
    int blocks_per_grid = (num_vectors + threads_per_block - 1) / threads_per_block;
    relu_kernel<<<blocks_per_grid, threads_per_block>>>(input, output, N);
    cudaDeviceSynchronize();
}
