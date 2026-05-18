#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int kBlockSize, int kItemsPerThread>
__global__ void add_vector_kernel(const float* A, const float* B, float* C, int N) {
    using BlockLoad = cub::BlockLoad<float, kBlockSize, kItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockStore =
        cub::BlockStore<float, kBlockSize, kItemsPerThread, cub::BLOCK_STORE_VECTORIZE>;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockStore::TempStorage store;
    } temp_storage;

    int block_offset = blockIdx.x * kBlockSize * kItemsPerThread;
    float a_items[kItemsPerThread];
    float b_items[kItemsPerThread];
    BlockLoad(temp_storage.load).Load(A + block_offset, a_items, N - block_offset);
    BlockLoad(temp_storage.load).Load(B + block_offset, b_items, N - block_offset);
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
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
