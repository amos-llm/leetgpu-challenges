#include <cuda_runtime.h>

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void silu_kernel(const float* input, float* output, int N) {
    int off = threadIdx.x + blockIdx.x * blockDim.x;
    if (off >= N) {
        return;
    }
    output[off] = input[off] * sigmoid(input[off]);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
