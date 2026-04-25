#include <cuda_runtime.h>

__device__ float gelu(float x) {
    return 0.5 * x * (1 + erf(x * 0.70710678118));
}

__global__ void geglu_kernel(const float* input, float* output, int halfN) {
    int off = threadIdx.x + blockIdx.x * blockDim.x;
    if (off >= halfN) {
        return;
    }
    float x1 = input[off];
    float x2 = input[off + halfN];
    output[off] = x1 * gelu(x2);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
