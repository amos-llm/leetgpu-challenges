#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <float.h>

template <int kBlockSize, int kItemsPerThread>
__global__ void topk_kernel(const float* input, float* output, int N, int k) {
    using BlockLoad = cub::BlockLoad<float, kBlockSize, kItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockSort = cub::BlockRadixSort<float, kBlockSize, kItemsPerThread>;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockSort::TempStorage sort;
    } temp_storage;

    int block_offset = blockIdx.x * kBlockSize * kItemsPerThread;
    float items[kItemsPerThread];
    BlockLoad(temp_storage.load).Load(input + block_offset, items, N - block_offset, -FLT_MAX);
    BlockSort(temp_storage.sort).SortDescending(items);

#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        int idx = threadIdx.x * kItemsPerThread + i;
        if (idx < k) {
            output[blockIdx.x * k + idx] = items[i];
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    constexpr int kBlockSize = 256;
    constexpr int kItemsPerThread = 4;
    constexpr int kItemsPerBlock = kBlockSize * kItemsPerThread;
    int grid_size = (N + kItemsPerBlock - 1) / kItemsPerBlock;
    if (grid_size > 1) {
        float* d_block_topk;
        cudaMalloc(&d_block_topk, grid_size * k * sizeof(float));
        topk_kernel<kBlockSize, kItemsPerThread>
            <<<grid_size, kBlockSize>>>(input, d_block_topk, N, k);
        solve(d_block_topk, output, grid_size * k, k);
        cudaFree(d_block_topk);
    } else {
        topk_kernel<kBlockSize, kItemsPerThread><<<grid_size, kBlockSize>>>(input, output, N, k);
    }
    cudaDeviceSynchronize();
}
