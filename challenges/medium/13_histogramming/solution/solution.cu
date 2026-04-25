#include <cuda_runtime.h>

__global__ void hist(const int* input, int* histogram, int N, int num_bins) {
    extern __shared__ int smem[];

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        smem[i] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        int bin = input[i];
        atomicAdd(&smem[bin], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&histogram[i], smem[i]);
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;
    hist<<<num_blocks, block_size, num_bins * sizeof(int)>>>(input, histogram, N, num_bins);
    cudaDeviceSynchronize();
}
