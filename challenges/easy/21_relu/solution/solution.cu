#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int off = threadIdx.x + blockIdx.x * blockDim.x;
    if (off >= N) {
        return;
    }
    output[off] = fmax(0.0f, input[off]);
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
