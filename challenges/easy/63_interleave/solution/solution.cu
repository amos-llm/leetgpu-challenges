#include <cuda_runtime.h>

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N) {
    int off = threadIdx.x + blockIdx.x * blockDim.x;
    if (off >= N) {
        return;
    }
    output[off * 2] = A[off];
    output[off * 2 + 1] = B[off];
}

// A, B, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, N);
    cudaDeviceSynchronize();
}
