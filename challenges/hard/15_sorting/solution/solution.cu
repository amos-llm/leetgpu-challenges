#include <cub/cub.cuh>
#include <cuda_runtime.h>

// data is device pointer
extern "C" void solve(float* data, int N) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, data, data, N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, data, data, N);
    cudaDeviceSynchronize();
    cudaFree(d_temp_storage);
}
