#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int BlockSize, int ItemsPerThread>
__global__ void count_positive_kernel(const float* input, int* output, int N) {
    using BlockLoad = cub::BlockLoad<float, BlockSize, ItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockReduce = cub::BlockReduce<int, BlockSize>;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    float items[ItemsPerThread];
    int block_offset = blockIdx.x * BlockSize * ItemsPerThread;
    BlockLoad(temp_storage.load).Load(input + block_offset, items, N - block_offset, 0.0f);

    int count = 0;
#pragma unroll
    for (int i = 0; i < ItemsPerThread; ++i) {
        if (items[i] > 0.0f) {
            count++;
        }
    }

    int block_count = BlockReduce(temp_storage.reduce).Sum(count);
    if (threadIdx.x == 0) {
        output[blockIdx.x] = block_count;
    }
}

template <int BlockSize, int ItemsPerThread>
__global__ void compact_kernel(const float* input, float* output, int* offset, int N) {
    using BlockLoad = cub::BlockLoad<float, BlockSize, ItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockScan = cub::BlockScan<int, BlockSize>;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockScan::TempStorage scan;
    } temp_storage;

    float items[ItemsPerThread];
    int block_offset = blockIdx.x * BlockSize * ItemsPerThread;
    BlockLoad(temp_storage.load).Load(input + block_offset, items, N - block_offset, 0.0f);

    int flags[ItemsPerThread];
#pragma unroll
    for (int i = 0; i < ItemsPerThread; ++i) {
        if (items[i] > 0.0f) {
            flags[i] = 1;
        } else {
            flags[i] = 0;
        }
    }
    int local_offsets[ItemsPerThread];
    BlockScan(temp_storage.scan).ExclusiveSum(flags, local_offsets);

    int output_start = 0;
    if (blockIdx.x > 0) {
        output_start = offset[blockIdx.x - 1];
    }
#pragma unroll
    for (int i = 0; i < ItemsPerThread; ++i) {
        if (flags[i] == 1) {
            output[output_start + local_offsets[i]] = items[i];
        }
    }
}

// A, out are device pointers
extern "C" void solve(const float* A, int N, float* out) {
    constexpr int kBlockSize = 256;
    constexpr int kItemsPerThread = 4;
    constexpr int kItemsPerBlock = kBlockSize * kItemsPerThread;
    int grid_size = (N + kItemsPerBlock - 1) / kItemsPerBlock;

    int* d_positive;
    cudaMalloc(&d_positive, grid_size * sizeof(int));
    count_positive_kernel<kBlockSize, kItemsPerThread><<<grid_size, kBlockSize>>>(A, d_positive, N);

    int* d_prefix_sum;
    cudaMalloc(&d_prefix_sum, grid_size * sizeof(int));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_positive, d_prefix_sum,
                                  grid_size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_positive, d_prefix_sum,
                                  grid_size);

    compact_kernel<kBlockSize, kItemsPerThread><<<grid_size, kBlockSize>>>(A, out, d_prefix_sum, N);

    cudaDeviceSynchronize();
    cudaFree(d_temp_storage);
    cudaFree(d_prefix_sum);
    cudaFree(d_positive);
}
