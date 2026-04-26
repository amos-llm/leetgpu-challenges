#include <cub/cub.cuh>
#include <cuda_runtime.h>

__global__ void max_subarray_kernel(const int* input, int* output, int N, int window_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > N - window_size) {
        return;
    }

    int sum = input[idx + window_size - 1];
    if (idx > 0) {
        sum -= input[idx - 1];
    }
    atomicMax(output, sum);
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int window_size) {
    int int_min = INT_MIN;
    cudaMemcpy(output, &int_min, sizeof(int), cudaMemcpyHostToDevice);

    int* d_prefix_sum = nullptr;
    cudaMalloc(&d_prefix_sum, N * sizeof(int));
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, input, d_prefix_sum, N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, input, d_prefix_sum, N);

    constexpr int kBlockSize = 256;
    int grid_size = (N - window_size + 1 + kBlockSize - 1) / kBlockSize;
    max_subarray_kernel<<<grid_size, kBlockSize>>>(d_prefix_sum, output, N, window_size);

    cudaDeviceSynchronize();
    cudaFree(d_temp_storage);
    cudaFree(d_prefix_sum);
}
