#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int off = threadIdx.x + blockIdx.x * blockDim.x;
    if (off >= N / 2) {
        return;
    }
    float temp = input[N - off - 1];
    input[N - off - 1] = input[off];
    input[off] = temp;
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N / 2 + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
