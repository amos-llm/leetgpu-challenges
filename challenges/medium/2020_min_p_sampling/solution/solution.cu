#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <float.h>

template <int kBlockSize>
__global__ void min_p_sampling_kernel(const float* logits, float* probs, float min_p, int B,
                                      int V) {
    using BlockReduce = cub::BlockReduce<float, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int batch_idx = blockIdx.x;

    float max_logit = -FLT_MAX;
    for (int i = threadIdx.x; i < V; i += kBlockSize) {
        max_logit = fmaxf(max_logit, logits[batch_idx * V + i]);
    }
    max_logit = BlockReduce(temp_storage).Reduce(max_logit, cuda::maximum<float>());
    __shared__ float s_max_logit;
    if (threadIdx.x == 0) {
        s_max_logit = max_logit;
    }
    __syncthreads();

    float logit_sum = 0.0f;
    for (int i = threadIdx.x; i < V; i += kBlockSize) {
        logit_sum += logits[batch_idx * V + i];
    }
    logit_sum = BlockReduce(temp_storage).Sum(logit_sum);
    __shared__ float s_logit_sum;
    if (threadIdx.x == 0) {
        s_logit_sum = logit_sum;
    }
    __syncthreads();

    float valid_logit_sum = 0.0f;
    for (int i = threadIdx.x; i < V; i += kBlockSize) {
        float logit = expf(logits[batch_idx * V + i] - s_max_logit);
        if (logit >= min_p) {
            valid_logit_sum += logit;
        }
    }
    valid_logit_sum = BlockReduce(temp_storage).Sum(valid_logit_sum);
    __shared__ float s_valid_logit_sum;
    if (threadIdx.x == 0) {
        s_valid_logit_sum = valid_logit_sum;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < V; i += kBlockSize) {
        float logit = expf(logits[batch_idx * V + i] - s_max_logit);
        if (logit >= min_p) {
            probs[batch_idx * V + i] = logit / s_valid_logit_sum;
        } else {
            probs[batch_idx * V + i] = 0.0f;
        }
    }
}

// logits, probs are device pointers
extern "C" void solve(const float* logits, float* probs, float min_p, int B, int V) {
    constexpr int kBlockSize = 256;
    min_p_sampling_kernel<kBlockSize><<<B, kBlockSize>>>(logits, probs, min_p, B, V);
    cudaDeviceSynchronize();
}
