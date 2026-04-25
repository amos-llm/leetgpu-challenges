#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int BlockSize, int ItemsPerThread>
__global__ void softmax_kernel(const float* input, float* output, int N) {
    struct RowStats {
        float max;
        float sum;
    };

    using BlockLoad = cub::BlockLoad<float, BlockSize, ItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockStore =
        cub::BlockStore<float, BlockSize, ItemsPerThread, cub::BLOCK_STORE_VECTORIZE>;
    using BlockReduce = cub::BlockReduce<RowStats, BlockSize>;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockStore::TempStorage store;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    RowStats stats = {-1e38f, 0.0f};
    float items[ItemsPerThread];

    for (int offset = 0; offset < N; offset += BlockSize * ItemsPerThread) {
        BlockLoad(temp_storage.load).Load(input + offset, items, N - offset);
#pragma unroll
        for (int i = 0; i < ItemsPerThread; ++i) {
            int idx = offset + threadIdx.x * ItemsPerThread + i;
            if (idx < N) {
                float val = items[i];
                float max_new = fmaxf(stats.max, val);
                stats.sum = stats.sum * __expf(stats.max - max_new) + __expf(val - max_new);
                stats.max = max_new;
            }
        }
    }

    struct ReduceOp {
        __device__ __forceinline__ RowStats operator()(const RowStats& a, const RowStats& b) {
            float max_new = fmaxf(a.max, b.max);
            return {max_new, a.sum * __expf(a.max - max_new) + b.sum * __expf(b.max - max_new)};
        }
    };

    RowStats reduced_stats = BlockReduce(temp_storage.reduce).Reduce(stats, ReduceOp());

    __shared__ float row_max;
    __shared__ float row_sum;

    if (threadIdx.x == 0) {
        row_max = reduced_stats.max;
        row_sum = reduced_stats.sum;
    }
    __syncthreads();

    for (int offset = 0; offset < N; offset += BlockSize * ItemsPerThread) {
        BlockLoad(temp_storage.load).Load(input + offset, items, N - offset);
#pragma unroll
        for (int i = 0; i < ItemsPerThread; ++i) {
            items[i] = __expf(items[i] - row_max) / row_sum;
        }
        BlockStore(temp_storage.store).Store(output + offset, items, N - offset);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    constexpr int kBlockSize = 256;
    constexpr int kItemsPerThread = 4;
    softmax_kernel<kBlockSize, kItemsPerThread><<<1, kBlockSize>>>(input, output, N);
    cudaDeviceSynchronize();
}
