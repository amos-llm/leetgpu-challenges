#include <cuda_runtime.h>

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void silu_kernel(const float* input, float* output, int N) {
    int vec_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx = vec_idx * 4;
    if (global_idx + 3 < N) {
        float4 val = *reinterpret_cast<const float4*>(&input[global_idx]);
        val.x = val.x * sigmoid(val.x);
        val.y = val.y * sigmoid(val.y);
        val.z = val.z * sigmoid(val.z);
        val.w = val.w * sigmoid(val.w);
        *reinterpret_cast<float4*>(&output[global_idx]) = val;
    } else {
        for (int i = 0; global_idx + i < N; ++i) {
            float x = input[global_idx + i];
            output[global_idx + i] = x * sigmoid(x);
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int num_vectors = (N + 3) / 4;
    int threads_per_block = 256;
    int blocks_per_grid = (num_vectors + threads_per_block - 1) / threads_per_block;
    silu_kernel<<<blocks_per_grid, threads_per_block>>>(input, output, N);
    cudaDeviceSynchronize();
}
