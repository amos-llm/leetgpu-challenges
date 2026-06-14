#include <cfloat>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

struct LogitId {
    float logit;
    int id;
};

struct ReduceOp {
    __device__ __forceinline__ LogitId operator()(const LogitId& a, const LogitId& b) {
        if (a.logit != b.logit) {
            return a.logit > b.logit ? a : b;
        }
        return a.id < b.id ? a : b;
    }
};

template <int kBlockSize>
__global__ void greedy_decoding_kernel(const float* logits, int* tokens, int vocab_size) {
    using BlockReduce = cub::BlockReduce<LogitId, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int batch_idx = blockIdx.x;
    LogitId local_max{-FLT_MAX, -1};
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float logit = logits[batch_idx * vocab_size + i];
        if (logit > local_max.logit) {
            local_max.logit = logit;
            local_max.id = i;
        }
    }
    local_max = BlockReduce(temp_storage).Reduce(local_max, ReduceOp());
    if (threadIdx.x == 0) {
        tokens[batch_idx] = local_max.id;
    }
}

// logits, tokens are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* logits, int* tokens, int batch_size, int vocab_size) {
    const int kBlockSize = 256;
    greedy_decoding_kernel<kBlockSize><<<batch_size, kBlockSize>>>(logits, tokens, vocab_size);
    cudaDeviceSynchronize();
}
