#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int BlockSize, int ItemsPerThread>
__global__ void monte_carlo_kernel(const float* y_samples, float* result, float a, float b,
                                   int n_samples) {
    using BlockLoad = cub::BlockLoad<float, BlockSize, ItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockReduce = cub::BlockReduce<float, BlockSize>;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    float items[ItemsPerThread];
    int block_offset = blockIdx.x * BlockSize * ItemsPerThread;
    BlockLoad(temp_storage.load)
        .Load(y_samples + block_offset, items, n_samples - block_offset, 0.0f);

    float sum = BlockReduce(temp_storage.reduce).Sum(items);
    if (threadIdx.x == 0) {
        atomicAdd(result, (b - a) * sum / n_samples);
    }
}

// y_samples, result are device pointers
extern "C" void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
    constexpr int kBlockSize = 256;
    constexpr int kItemsPerThread = 4;
    constexpr int kItemsPerBlock = kBlockSize * kItemsPerThread;
    int grid_size = (n_samples + kItemsPerBlock - 1) / kItemsPerBlock;
    monte_carlo_kernel<kBlockSize, kItemsPerThread>
        <<<grid_size, kBlockSize>>>(y_samples, result, a, b, n_samples);
    cudaDeviceSynchronize();
}
