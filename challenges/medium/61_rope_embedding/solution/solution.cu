#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int BlockSize, int ItemsPerThread>
__global__ void rope_kernel(float* Q, float* cos, float* sin, float* output, int M, int D) {
    using BlockLoad = cub::BlockLoad<float, BlockSize, ItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockStore =
        cub::BlockStore<float, BlockSize, ItemsPerThread, cub::BLOCK_STORE_VECTORIZE>;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockStore::TempStorage store;
    } temp_storage;

    float* Q_base = Q + blockIdx.x * D;
    float* cos_base = cos + blockIdx.x * D;
    float* sin_base = sin + blockIdx.x * D;
    float* output_base = output + blockIdx.x * D;

    float q1_items[ItemsPerThread];
    float q2_items[ItemsPerThread];
    float c1_items[ItemsPerThread];
    float c2_items[ItemsPerThread];
    float s1_items[ItemsPerThread];
    float s2_items[ItemsPerThread];
    int block_offset = blockIdx.y * BlockSize * ItemsPerThread;
    BlockLoad(temp_storage.load).Load(Q_base + block_offset, q1_items, D / 2 - block_offset);
    BlockLoad(temp_storage.load)
        .Load(Q_base + block_offset + D / 2, q2_items, D / 2 - block_offset);
    BlockLoad(temp_storage.load).Load(cos_base + block_offset, c1_items, D / 2 - block_offset);
    BlockLoad(temp_storage.load)
        .Load(cos_base + block_offset + D / 2, c2_items, D / 2 - block_offset);
    BlockLoad(temp_storage.load).Load(sin_base + block_offset, s1_items, D / 2 - block_offset);
    BlockLoad(temp_storage.load)
        .Load(sin_base + block_offset + D / 2, s2_items, D / 2 - block_offset);

    float q_items[ItemsPerThread];
#pragma unroll
    for (int i = 0; i < ItemsPerThread; i++) {
        q_items[i] = q1_items[i] * c1_items[i] - q2_items[i] * s1_items[i];
    }
    BlockStore(temp_storage.store).Store(output_base + block_offset, q_items, D / 2 - block_offset);
#pragma unroll
    for (int i = 0; i < ItemsPerThread; i++) {
        q_items[i] = q2_items[i] * c2_items[i] + q1_items[i] * s2_items[i];
    }
    BlockStore(temp_storage.store)
        .Store(output_base + block_offset + D / 2, q_items, D / 2 - block_offset);
}

// Q, cos, sin, output are device pointers
extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {
    constexpr int kBlockSize = 256;
    constexpr int kItemsPerThread = 4;
    constexpr int kItemsPerBlock = kBlockSize * kItemsPerThread;
    dim3 grid_size(M, (D / 2 + kItemsPerBlock - 1) / kItemsPerBlock);
    rope_kernel<kBlockSize, kItemsPerThread><<<grid_size, kBlockSize>>>(Q, cos, sin, output, M, D);
    cudaDeviceSynchronize();
}
