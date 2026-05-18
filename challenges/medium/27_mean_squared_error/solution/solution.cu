#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int kBlockSize, int kItemsPerThread>
__global__ void mse_kernel(const float* predictions, const float* targets, float* mse, int N) {
    using BlockLoad = cub::BlockLoad<float, kBlockSize, kItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockReduce = cub::BlockReduce<float, kBlockSize>;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    float p_items[kItemsPerThread];
    float t_items[kItemsPerThread];
    int block_offset = blockIdx.x * kBlockSize * kItemsPerThread;
    BlockLoad(temp_storage.load).Load(predictions + block_offset, p_items, N - block_offset, 0.0f);
    BlockLoad(temp_storage.load).Load(targets + block_offset, t_items, N - block_offset, 0.0f);
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
        p_items[i] = (p_items[i] - t_items[i]) * (p_items[i] - t_items[i]);
    }

    float sum = BlockReduce(temp_storage.reduce).Sum(p_items);
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
