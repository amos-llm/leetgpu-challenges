#include <cuda_runtime.h>
#include <math.h>

__device__ float sigmoid(float x) {
    return 1 / (1 + expf(-x));
}

__global__ void sigmoid_kernel(const float* X, float* Y, int N) {
    int off = threadIdx.x + blockIdx.x * blockDim.x;
    if (off >= N) {
        return;
    }
    Y[off] = sigmoid(X[off]);
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
