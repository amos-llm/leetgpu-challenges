#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int BlockSize, int ItemsPerThread>
__global__ void mse_kernel(const float* predictions, const float* targets, float* mse, int N) {
    using BLockLoad = cub::BlockLoad<float, BlockSize, ItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BLockReduce = cub::BlockReduce<float, BlockSize>;

    __shared__ union {
        typename BLockLoad::TempStorage load;
        typename BLockReduce::TempStorage reduce;
    } temp_storage;

    float p_items[ItemsPerThread];
    float t_items[ItemsPerThread];
    int block_offset = blockIdx.x * BlockSize * ItemsPerThread;
    BLockLoad(temp_storage.load).Load(predictions + block_offset, p_items, N - block_offset, 0.0f);
    BLockLoad(temp_storage.load).Load(targets + block_offset, t_items, N - block_offset, 0.0f);
#pragma unroll
    for (int i = 0; i < ItemsPerThread; i++) {
        p_items[i] = (p_items[i] - t_items[i]) * (p_items[i] - t_items[i]);
    }

    float sum = BLockReduce(temp_storage.reduce).Sum(p_items);
    if (threadIdx.x == 0) {
        atomicAdd(mse, sum / N);
    }
}

// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    constexpr int kBlockSize = 256;
    constexpr int kItemsPerThread = 4;
    constexpr int kItemsPerBlock = kBlockSize * kItemsPerThread;
    int grid_size = (N + kItemsPerBlock - 1) / kItemsPerBlock;
    mse_kernel<kBlockSize, kItemsPerThread>
        <<<grid_size, kBlockSize>>>(predictions, targets, mse, N);
    cudaDeviceSynchronize();
}
