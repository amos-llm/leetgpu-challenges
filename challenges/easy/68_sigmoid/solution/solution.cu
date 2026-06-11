#include <cuda_runtime.h>
#include <math.h>

__device__ float sigmoid(float x) {
    return 1 / (1 + expf(-x));
}

__global__ void sigmoid_kernel(const float* X, float* Y, int N) {
    int vec_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx = vec_idx * 4;
    if (global_idx + 3 < N) {
        float4 val = *reinterpret_cast<const float4*>(&X[global_idx]);
        val.x = sigmoid(val.x);
        val.y = sigmoid(val.y);
        val.z = sigmoid(val.z);
        val.w = sigmoid(val.w);
        *reinterpret_cast<float4*>(&Y[global_idx]) = val;
    } else {
        for (int i = 0; global_idx + i < N; ++i) {
            Y[global_idx + i] = sigmoid(X[global_idx + i]);
        }
    }
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int num_vectors = (N + 3) / 4;
    int threads_per_block = 256;
    int blocks_per_grid = (num_vectors + threads_per_block - 1) / threads_per_block;
    sigmoid_kernel<<<blocks_per_grid, threads_per_block>>>(X, Y, N);
    cudaDeviceSynchronize();
}
