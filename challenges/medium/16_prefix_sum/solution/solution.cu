#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

template <int BlockSize, int ItemsPerThread>
__global__ void scan_block(const float* input, float* output, float* block_sum, int N) {
    using BlockLoad = cub::BlockLoad<float, BlockSize, ItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockStore =
        cub::BlockStore<float, BlockSize, ItemsPerThread, cub::BLOCK_STORE_VECTORIZE>;
    using BlockScan = cub::BlockScan<float, BlockSize>;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockStore::TempStorage store;
        typename BlockScan::TempStorage scan;
    } temp_storage;

    float items[ItemsPerThread];
    int block_offset = blockIdx.x * BlockSize * ItemsPerThread;
    BlockLoad(temp_storage.load).Load(input + block_offset, items, N - block_offset, 0.0f);
    float sum = 0.0f;
    BlockScan(temp_storage.scan).InclusiveScan(items, items, cuda::std::plus<float>(), sum);
    __syncthreads();
    BlockStore(temp_storage.store).Store(output + block_offset, items, N - block_offset);
    if (threadIdx.x == 0) {
        block_sum[blockIdx.x] = sum;
    }
}

template <int BlockSize, int ItemsPerThread>
__global__ void add_offset(const float* input, const float* offset, float* output, int N) {
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
    int bid = blockIdx.x;
    if (bid > 0) {
#pragma unroll
        for (int i = 0; i < ItemsPerThread; ++i) {
            items[i] += offset[bid - 1];
        }
    }
    BlockStore(temp_storage.store).Store(output + block_offset, items, N - block_offset);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    constexpr int kBlockSize = 256;
    constexpr int kItemsPerThread = 4;
    constexpr int kItemsPerBlock = kBlockSize * kItemsPerThread;
    int grid_size = (N + kItemsPerBlock - 1) / kItemsPerBlock;
    thrust::device_vector<float> d_block_sum(grid_size, 0.0f);
    float* d_block_sum_ptr = thrust::raw_pointer_cast(d_block_sum.data());
    scan_block<kBlockSize, kItemsPerThread>
        <<<grid_size, kBlockSize>>>(input, output, d_block_sum_ptr, N);

    if (grid_size > 1) {
        solve(d_block_sum_ptr, d_block_sum_ptr, grid_size);
    }

    add_offset<kBlockSize, kItemsPerThread>
        <<<grid_size, kBlockSize>>>(output, d_block_sum_ptr, output, N);
    cudaDeviceSynchronize();
}
