#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int BlockSize, int ItemsPerThread>
__global__ void sum(const int* input, int* output, int N, int S, int E) {
    using BlockLoad = cub::BlockLoad<int, BlockSize, ItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockReduce = cub::BlockReduce<int, BlockSize>;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    int items[ItemsPerThread];
    int block_offset = S + blockIdx.x * BlockSize * ItemsPerThread;
    BlockLoad(temp_storage.load).Load(input + block_offset, items, E - block_offset + 1, 0);

    int block_sum = BlockReduce(temp_storage.reduce).Sum(items);
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    constexpr int kBlockSize = 256;
    constexpr int kItemsPerThread = 4;
    constexpr int kItemsPerBlock = kBlockSize * kItemsPerThread;
    int grid_size = (E - S + 1 + kItemsPerBlock - 1) / kItemsPerBlock;
    sum<kBlockSize, kItemsPerThread><<<grid_size, kBlockSize>>>(input, output, N, S, E);
    cudaDeviceSynchronize();
}
