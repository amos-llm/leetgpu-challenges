#include <cuda_runtime.h>

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) {
        return;
    }
    float2 vec;
    vec.x = A[idx];
    vec.y = B[idx];
    reinterpret_cast<float2*>(output)[idx] = vec;
}

// A, B, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* output, int N) {
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    interleave_kernel<<<blocks_per_grid, threads_per_block>>>(A, B, output, N);
    cudaDeviceSynchronize();
}
