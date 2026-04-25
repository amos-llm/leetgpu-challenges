#include <cuda_runtime.h>

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float silu(float x) {
    return x * sigmoid(x);
}

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int off = threadIdx.x + blockIdx.x * blockDim.x;
    if (off >= halfN) {
        return;
    }
    float x1 = input[off];
    float x2 = input[off + halfN];
    output[off] = silu(x1) * x2;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
