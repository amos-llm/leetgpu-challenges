#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int kBlockSize, int kItemsPerThread>
__global__ void local_scan_kernel(const float* input, float* output, float* block_sum, int N) {
    using BlockLoad = cub::BlockLoad<float, kBlockSize, kItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockStore =
        cub::BlockStore<float, kBlockSize, kItemsPerThread, cub::BLOCK_STORE_VECTORIZE>;
    using BlockScan = cub::BlockScan<float, kBlockSize>;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockStore::TempStorage store;
        typename BlockScan::TempStorage scan;
    } temp_storage;

    float items[kItemsPerThread];
    int block_offset = blockIdx.x * kBlockSize * kItemsPerThread;
    BlockLoad(temp_storage.load).Load(input + block_offset, items, N - block_offset, 0.0f);
    float sum = 0.0f;
    BlockScan(temp_storage.scan).InclusiveScan(items, items, thrust::plus<float>(), sum);
    BlockStore(temp_storage.store).Store(output + block_offset, items, N - block_offset);
    if (threadIdx.x == 0) {
        block_sum[blockIdx.x] = sum;
    }
}

template <int kBlockSize, int kItemsPerThread>
__global__ void add_offset_kernel(const float* input, const float* offset, float* output, int N) {
    using BlockLoad = cub::BlockLoad<float, kBlockSize, kItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockStore =
        cub::BlockStore<float, kBlockSize, kItemsPerThread, cub::BLOCK_STORE_VECTORIZE>;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockStore::TempStorage store;
    } temp_storage;

    float items[kItemsPerThread];
    int block_offset = blockIdx.x * kBlockSize * kItemsPerThread;
    BlockLoad(temp_storage.load).Load(input + block_offset, items, N - block_offset, 0.0f);
    int bid = blockIdx.x;
    if (bid > 0) {
#pragma unroll
        for (int i = 0; i < kItemsPerThread; ++i) {
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
    float* d_block_sum;
    cudaMalloc(&d_block_sum, grid_size * sizeof(float));
    local_scan_kernel<kBlockSize, kItemsPerThread>
        <<<grid_size, kBlockSize>>>(input, output, d_block_sum, N);
    if (grid_size > 1) {
        solve(d_block_sum, d_block_sum, grid_size);
    }
    add_offset_kernel<kBlockSize, kItemsPerThread>
        <<<grid_size, kBlockSize>>>(output, d_block_sum, output, N);
    cudaDeviceSynchronize();
    cudaFree(d_block_sum);
}
