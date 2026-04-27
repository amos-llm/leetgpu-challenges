#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <float.h>

template <int BlockSize, int ItemsPerThread>
__global__ void moe_topk_kernel(const float* logits, float* topk_weights, int* topk_indices, int M,
                                int E, int k) {
    using BlockSort = cub::BlockRadixSort<float, BlockSize, ItemsPerThread, int>;
    using WarpReduce = cub::WarpReduce<float>;

    __shared__ union {
        typename BlockSort::TempStorage sort;
        typename WarpReduce::TempStorage reduce;
    } temp_storage;

    float logit_items[ItemsPerThread];
    int index_items[ItemsPerThread];
    if (threadIdx.x < E) {
        logit_items[0] = logits[blockIdx.x * E + threadIdx.x];
    } else {
        logit_items[0] = -FLT_MAX;
    }
    index_items[0] = threadIdx.x;
    BlockSort(temp_storage.sort).SortDescending(logit_items, index_items);

    float weight = 0;
    if (threadIdx.x < k) {
        weight = expf(logit_items[0]);
    }
    __syncthreads();

    __shared__ float weight_sum;
    float block_sum = WarpReduce(temp_storage.reduce).Sum(weight);
    if (threadIdx.x == 0) {
        weight_sum = block_sum;
    }
    __syncthreads();

    if (threadIdx.x < k) {
        topk_weights[blockIdx.x * k + threadIdx.x] = weight / weight_sum;
        topk_indices[blockIdx.x * k + threadIdx.x] = index_items[0];
    }
}

// logits, topk_weights, topk_indices are device pointers
extern "C" void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k) {
    constexpr int kBlockSize = 256;
    moe_topk_kernel<kBlockSize, 1><<<M, kBlockSize>>>(logits, topk_weights, topk_indices, M, E, k);
    cudaDeviceSynchronize();
}
