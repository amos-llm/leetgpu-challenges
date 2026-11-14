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
__global__ void row_argmax_kernel(const float* logits, int* output, int V) {
    using BlockReduce = cub::BlockReduce<LogitId, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int batch_idx = blockIdx.x;
    LogitId local_max{-FLT_MAX, -1};
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float logit = logits[batch_idx * V + i];
        if (logit > local_max.logit) {
            local_max.logit = logit;
            local_max.id = i;
        }
    }
    local_max = BlockReduce(temp_storage).Reduce(local_max, ReduceOp());
    if (threadIdx.x == 0) {
        output[batch_idx] = local_max.id;
    }
}

// logits, output are device pointers
extern "C" void solve(const float* logits, int* output, int B, int V) {
    const int kBlockSize = 256;
    row_argmax_kernel<kBlockSize><<<B, kBlockSize>>>(logits, output, V);
    cudaDeviceSynchronize();
}
