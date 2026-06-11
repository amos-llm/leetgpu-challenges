#include <cuda_runtime.h>

__device__ float gelu(float x) {
    return 0.5 * x * (1 + erf(x * 0.70710678118));
}

__global__ void geglu_kernel(const float* input, float* output, int halfN) {
    int vec_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx = vec_idx * 4;
    if (global_idx + 3 < halfN) {
        float4 x1 = *reinterpret_cast<const float4*>(&input[global_idx]);
        float4 x2 = *reinterpret_cast<const float4*>(&input[global_idx + halfN]);
        float4 out;
        out.x = x1.x * gelu(x2.x);
        out.y = x1.y * gelu(x2.y);
        out.z = x1.z * gelu(x2.z);
        out.w = x1.w * gelu(x2.w);
        *reinterpret_cast<float4*>(&output[global_idx]) = out;
    } else {
        for (int i = 0; global_idx + i < halfN; ++i) {
            float a = input[global_idx + i];
            float b = input[global_idx + i + halfN];
            output[global_idx + i] = a * gelu(b);
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int num_vectors = (halfN + 3) / 4;
    int threads_per_block = 256;
    int blocks_per_grid = (num_vectors + threads_per_block - 1) / threads_per_block;
    geglu_kernel<<<blocks_per_grid, threads_per_block>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
