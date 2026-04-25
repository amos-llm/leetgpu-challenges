#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int BlockSize, int ItemsPerThread>
__global__ void sum(const int* input, int* output, int N, int M, int K, int S_DEP, int E_DEP,
                    int S_ROW, int E_ROW, int S_COL, int E_COL) {
    using BlockLoad = cub::BlockLoad<int, BlockSize, ItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockReduce = cub::BlockReduce<int, BlockSize>;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    int items[ItemsPerThread];
    int block_offset = S_COL + blockIdx.z * BlockSize * ItemsPerThread;
    BlockLoad(temp_storage.load)
        .Load(input + (S_DEP + blockIdx.x) * M * K + (S_ROW + blockIdx.y) * K + block_offset, items,
              E_COL - block_offset + 1, 0);

    int block_sum = BlockReduce(temp_storage.reduce).Sum(items);
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K, int S_DEP, int E_DEP,
                      int S_ROW, int E_ROW, int S_COL, int E_COL) {
    constexpr int kBlockSize = 256;
    constexpr int kItemsPerThread = 4;
    constexpr int kItemsPerBlock = kBlockSize * kItemsPerThread;
    dim3 grid_size(E_DEP - S_DEP + 1, E_ROW - S_ROW + 1,
                   (E_COL - S_COL + 1 + kItemsPerBlock - 1) / kItemsPerBlock);
    sum<kBlockSize, kItemsPerThread><<<grid_size, kBlockSize>>>(input, output, N, M, K, S_DEP,
                                                                E_DEP, S_ROW, E_ROW, S_COL, E_COL);
    cudaDeviceSynchronize();
}
