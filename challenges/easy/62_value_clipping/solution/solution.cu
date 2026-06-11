#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int N) {
    int vec_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx = vec_idx * 4;
    if (global_idx + 3 < N) {
        float4 val = *reinterpret_cast<const float4*>(&input[global_idx]);
        val.x = fmin(hi, fmax(lo, val.x));
        val.y = fmin(hi, fmax(lo, val.y));
        val.z = fmin(hi, fmax(lo, val.z));
        val.w = fmin(hi, fmax(lo, val.w));
        *reinterpret_cast<float4*>(&output[global_idx]) = val;
    } else {
        for (int i = 0; global_idx + i < N; ++i) {
            output[global_idx + i] = fmin(hi, fmax(lo, input[global_idx + i]));
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int num_vectors = (N + 3) / 4;
    int threads_per_block = 256;
    int blocks_per_grid = (num_vectors + threads_per_block - 1) / threads_per_block;
    clip_kernel<<<blocks_per_grid, threads_per_block>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
