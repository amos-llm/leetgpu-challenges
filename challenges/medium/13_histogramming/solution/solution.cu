#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int BlockSize, int ItemsPerThread>
__global__ void histogram_kernel(const int* input, int* histogram, int N, int num_bins) {
    using BlockLoad = cub::BlockLoad<int, BlockSize, ItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    __shared__ typename BlockLoad::TempStorage temp_storage;
    extern __shared__ int s_histogram[];

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        s_histogram[i] = 0;
    }
    __syncthreads();

    int block_offset = blockIdx.x * BlockSize * ItemsPerThread;
    int items[ItemsPerThread];
    BlockLoad(temp_storage).Load(input + block_offset, items, N - block_offset, -1);
#pragma unroll
    for (int i = 0; i < ItemsPerThread; ++i) {
        int value = items[i];
        if (value >= 0 && value < num_bins) {
            atomicAdd(&s_histogram[value], 1);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&histogram[i], s_histogram[i]);
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    constexpr int kBlockSize = 512;
    constexpr int kItemsPerThread = 4;
    constexpr int kItemsPerBlock = kBlockSize * kItemsPerThread;
    int grid_size = (N + kItemsPerBlock - 1) / kItemsPerBlock;
    histogram_kernel<kBlockSize, kItemsPerThread>
        <<<grid_size, kBlockSize, num_bins * sizeof(int)>>>(input, histogram, N, num_bins);
    cudaDeviceSynchronize();
}
