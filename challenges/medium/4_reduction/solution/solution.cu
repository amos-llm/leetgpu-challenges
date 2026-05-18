#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int kBlockSize, int kItemsPerThread>
__global__ void sum_kernel(const float* input, float* output, int N) {
    using BlockLoad = cub::BlockLoad<float, kBlockSize, kItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockReduce = cub::BlockReduce<float, kBlockSize>;
    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    float items[kItemsPerThread];
    int block_offset = blockIdx.x * kBlockSize * kItemsPerThread;
    BlockLoad(temp_storage.load).Load(input + block_offset, items, N - block_offset, 0.0f);

    float block_sum = BlockReduce(temp_storage.reduce).Sum(items);
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    constexpr int kBlockSize = 256;
    constexpr int kItemsPerThread = 4;
    constexpr int kItemsPerBlock = kBlockSize * kItemsPerThread;
    const int grid_size = (N + kItemsPerBlock - 1) / kItemsPerBlock;
    sum_kernel<kBlockSize, kItemsPerThread><<<grid_size, kBlockSize>>>(input, output, N);
    cudaDeviceSynchronize();
}
