#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int BlockSize, int ItemsPerThread>
__global__ void count_kernel(const int* input, int* output, int N, int K) {
    using BLockLoad = cub::BlockLoad<int, BlockSize, ItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockReduce = cub::BlockReduce<int, BlockSize>;

    __shared__ union {
        typename BLockLoad::TempStorage load;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    int items[ItemsPerThread];
    int block_offset = blockIdx.x * BlockSize * ItemsPerThread;
    BLockLoad(temp_storage.load).Load(input + block_offset, items, N - block_offset, 0);

#pragma unroll
    for (int i = 0; i < ItemsPerThread; ++i) {
        items[i] = (items[i] == K) ? 1 : 0;
    }

    int count = BlockReduce(temp_storage.reduce).Sum(items);
    if (threadIdx.x == 0) {
        atomicAdd(output, count);
    }
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int K) {
    constexpr int kBlockSize = 256;
    constexpr int kItemsPerThread = 4;
    constexpr int kItemsPerBlock = kBlockSize * kItemsPerThread;
    int grid_size = (N + kItemsPerBlock - 1) / kItemsPerBlock;
    count_kernel<kBlockSize, kItemsPerThread><<<grid_size, kBlockSize>>>(input, output, N, K);
}
