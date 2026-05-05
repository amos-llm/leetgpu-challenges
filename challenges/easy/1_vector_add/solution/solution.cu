#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int BlockSize, int ItemsPerThread>
__global__ void add_vector_kernel(const float* A, const float* B, float* C, int N) {
    using BlockLoad = cub::BlockLoad<float, BlockSize, ItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockStore =
        cub::BlockStore<float, BlockSize, ItemsPerThread, cub::BLOCK_STORE_VECTORIZE>;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockStore::TempStorage store;
    } temp_storage;

    int block_offset = blockIdx.x * BlockSize * ItemsPerThread;
    float a_items[ItemsPerThread];
    float b_items[ItemsPerThread];
    BlockLoad(temp_storage.load).Load(A + block_offset, a_items, N - block_offset);
    BlockLoad(temp_storage.load).Load(B + block_offset, b_items, N - block_offset);
#pragma unroll
    for (int i = 0; i < ItemsPerThread; ++i) {
        a_items[i] += b_items[i];
    }
    BlockStore(temp_storage.store).Store(C + block_offset, a_items, N - block_offset);
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    constexpr int kBlockSize = 256;
    constexpr int kItemsPerThread = 4;
    constexpr int kItemsPerBlock = kBlockSize * kItemsPerThread;
    int grid_size = (N + kItemsPerBlock - 1) / kItemsPerBlock;
    add_vector_kernel<kBlockSize, kItemsPerThread><<<grid_size, kBlockSize>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
