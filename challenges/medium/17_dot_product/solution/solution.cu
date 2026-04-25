#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int BlockSize, int ItemsPerThread>
__global__ void dot_product(const float* A, const float* B, float* result, int N) {
    using BlockLoad = cub::BlockLoad<float, BlockSize, ItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockReduce = cub::BlockReduce<float, BlockSize>;
    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    float a_items[ItemsPerThread];
    float b_items[ItemsPerThread];
    int block_offset = blockIdx.x * BlockSize * ItemsPerThread;
    BlockLoad(temp_storage.load).Load(A + block_offset, a_items, N - block_offset, 0.0f);
    BlockLoad(temp_storage.load).Load(B + block_offset, b_items, N - block_offset, 0.0f);
#pragma unroll
    for (int i = 0; i < ItemsPerThread; ++i) {
        a_items[i] *= b_items[i];
    }

    float block_sum = BlockReduce(temp_storage.reduce).Sum(a_items);
    if (threadIdx.x == 0) {
        atomicAdd(result, block_sum);
    }
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    constexpr int kBlockSize = 256;
    constexpr int kItemsPerThread = 4;
    constexpr int kItemsPerBlock = kBlockSize * kItemsPerThread;
    int grid_size = (N + kItemsPerBlock - 1) / kItemsPerBlock;
    dot_product<kBlockSize, kItemsPerThread><<<grid_size, kBlockSize>>>(A, B, result, N);
    cudaDeviceSynchronize();
}
