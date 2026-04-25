#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int BlockSize, int ItemsPerThread>
__global__ void cross_entropy_kernel(const float* logits, const int* true_labels, float* loss,
                                     int N, int C) {
    struct RowStats {
        float max;
        float sum;
    };
    struct ReduceOp {
        __device__ __forceinline__ RowStats operator()(const RowStats& a, const RowStats& b) {
            float max = fmaxf(a.max, b.max);
            return RowStats{max, a.sum * expf(a.max - max) + b.sum * expf(b.max - max)};
        }
    };
    using BlockLoad = cub::BlockLoad<float, BlockSize, ItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockReduce = cub::BlockReduce<RowStats, BlockSize>;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    RowStats stats{-1e38f, 0.0f};
    float logit_items[ItemsPerThread];
    for (int i = 0; i < C; i += BlockSize * ItemsPerThread) {
        int block_offset = blockIdx.x * C + i;
        BlockLoad(temp_storage.load).Load(logits + block_offset, logit_items, C - i, -1e38f);

#pragma unroll
        for (int j = 0; j < ItemsPerThread; j++) {
            float max = fmaxf(stats.max, logit_items[j]);
            stats.sum = stats.sum * expf(stats.max - max) + expf(logit_items[j] - max);
            stats.max = max;
        }
    }

    RowStats reduced_stats = BlockReduce(temp_storage.reduce).Reduce(stats, ReduceOp());
    if (threadIdx.x == 0) {
        float true_logit = logits[blockIdx.x * C + true_labels[blockIdx.x]];
        atomicAdd(loss, (logf(reduced_stats.sum) - true_logit + reduced_stats.max) / N);
    }
}

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    constexpr int kBlockSize = 256;
    constexpr int kItemsPerThread = 4;
    dim3 grid_size(N, 1);
    cross_entropy_kernel<kBlockSize, kItemsPerThread>
        <<<grid_size, kBlockSize>>>(logits, true_labels, loss, N, C);
    cudaDeviceSynchronize();
}
