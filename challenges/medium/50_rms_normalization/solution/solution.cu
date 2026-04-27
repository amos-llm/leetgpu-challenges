#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int BlockSize, int ItemsPerThread>
__global__ void mean_square_kernel(const float* input, float* ms, int N) {
    using BlockLoad = cub::BlockLoad<float, BlockSize, ItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockReduce = cub::BlockReduce<float, BlockSize>;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    float items[ItemsPerThread];
    int block_offset = blockIdx.x * BlockSize * ItemsPerThread;
    BlockLoad(temp_storage.load).Load(input + block_offset, items, N - block_offset, 0.0f);

#pragma unroll
    for (int i = 0; i < ItemsPerThread; ++i) {
        items[i] = items[i] * items[i];
    }
    float sum_square = BlockReduce(temp_storage.reduce).Sum(items);
    if (threadIdx.x == 0) {
        atomicAdd(ms, sum_square / N);
    }
}

template <int BlockSize, int ItemsPerThread>
__global__ void norm_kernel(const float* input, float gamma, float beta, float* ms, float* output,
                            int N, float eps) {
    using BlockLoad = cub::BlockLoad<float, BlockSize, ItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockStore =
        cub::BlockStore<float, BlockSize, ItemsPerThread, cub::BLOCK_STORE_VECTORIZE>;
    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockStore::TempStorage store;
    } temp_storage;

    float items[ItemsPerThread];
    int block_offset = blockIdx.x * BlockSize * ItemsPerThread;
    BlockLoad(temp_storage.load).Load(input + block_offset, items, N - block_offset, 0.0f);

    float rrms = rsqrtf(*ms + eps);
#pragma unroll
    for (int i = 0; i < ItemsPerThread; ++i) {
        items[i] = items[i] * rrms * gamma + beta;
    }

    BlockStore(temp_storage.store).Store(output + block_offset, items, N - block_offset);
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {
    constexpr int kBlockSize = 256;
    constexpr int kItemsPerThread = 4;
    constexpr int kItemsPerBlock = kBlockSize * kItemsPerThread;
    int grid_size = (N + kItemsPerBlock - 1) / kItemsPerBlock;
    float* d_ms = nullptr;
    cudaMalloc(&d_ms, sizeof(float));
    cudaMemset(d_ms, 0, sizeof(float));
    mean_square_kernel<kBlockSize, kItemsPerThread><<<grid_size, kBlockSize>>>(input, d_ms, N);
    norm_kernel<kBlockSize, kItemsPerThread>
        <<<grid_size, kBlockSize>>>(input, gamma, beta, d_ms, output, N, eps);

    cudaDeviceSynchronize();
    cudaFree(d_ms);
}
