#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

template <int BlockSize, int ItemsPerThread>
__global__ void dot_product(const half* A, const half* B, float* result, int N) {
    using BlockLoad = cub::BlockLoad<half, BlockSize, ItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockReduce = cub::BlockReduce<float, BlockSize>;
    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    half a_items[ItemsPerThread];
    half b_items[ItemsPerThread];
    int block_offset = blockIdx.x * BlockSize * ItemsPerThread;
    BlockLoad(temp_storage.load).Load(A + block_offset, a_items, N - block_offset, half(0.0f));
    BlockLoad(temp_storage.load).Load(B + block_offset, b_items, N - block_offset, half(0.0f));

    float products[ItemsPerThread];
#pragma unroll
    for (int i = 0; i < ItemsPerThread; ++i) {
        products[i] = __half2float(a_items[i]) * __half2float(b_items[i]);
    }

    float block_sum = BlockReduce(temp_storage.reduce).Sum(products);
    if (threadIdx.x == 0) {
        atomicAdd(result, block_sum);
    }
}

__global__ void float2half(const float* input, half* output) {
    *output = __float2half(*input);
}

// A, B, result are device pointers
extern "C" void solve(const half* A, const half* B, half* result, int N) {
    constexpr int kBlockSize = 256;
    constexpr int kItemsPerThread = 4;
    constexpr int kItemsPerBlock = kBlockSize * kItemsPerThread;
    int grid_size = (N + kItemsPerBlock - 1) / kItemsPerBlock;
    thrust::device_vector<float> result_fp32(1, 0.0f);
    dot_product<kBlockSize, kItemsPerThread>
        <<<grid_size, kBlockSize>>>(A, B, thrust::raw_pointer_cast(result_fp32.data()), N);
    float2half<<<1, 1>>>(thrust::raw_pointer_cast(result_fp32.data()), result);
    cudaDeviceSynchronize();
}
