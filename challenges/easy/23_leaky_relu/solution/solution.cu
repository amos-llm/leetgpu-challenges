#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    int vec_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx = vec_idx * 4;
    if (global_idx + 3 < N) {
        float4 val = *reinterpret_cast<const float4*>(&input[global_idx]);
        val.x = val.x >= 0.0f ? val.x : 0.01f * val.x;
        val.y = val.y >= 0.0f ? val.y : 0.01f * val.y;
        val.z = val.z >= 0.0f ? val.z : 0.01f * val.z;
        val.w = val.w >= 0.0f ? val.w : 0.01f * val.w;
        *reinterpret_cast<float4*>(&output[global_idx]) = val;
    } else {
        for (int i = 0; global_idx + i < N; ++i) {
            float v = input[global_idx + i];
            output[global_idx + i] = v >= 0.0f ? v : 0.01f * v;
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int num_vectors = (N + 3) / 4;
    int threads_per_block = 256;
    int blocks_per_grid = (num_vectors + threads_per_block - 1) / threads_per_block;
    leaky_relu_kernel<<<blocks_per_grid, threads_per_block>>>(input, output, N);
    cudaDeviceSynchronize();
}
