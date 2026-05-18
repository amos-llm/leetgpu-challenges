#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/functional.h>

template <int kBlockSize, int kItemsPerThread>
__global__ void verify_kernel(const int* draft_tokens,      // [B, T]
                              const float* draft_probs,     // [B, T, V]
                              const float* target_probs,    // [B, T, V]
                              const float* uniform_samples, // [B, T + 1]
                              int* output_tokens,           // [B, T + 1]
                              int B, int T, int V) {
    __shared__ int num_accepted_tokens;
    int batch_idx = blockIdx.x;
    if (threadIdx.x == 0) {
        for (int t = 0; t < T; ++t) {
            if (t == 0) {
                num_accepted_tokens = 0;
            }
            int draft_token = draft_tokens[batch_idx * T + t];
            float draft_prob = draft_probs[batch_idx * T * V + t * V + draft_token];
            float target_prob = target_probs[batch_idx * T * V + t * V + draft_token];
            float alpha = fminf(1.0f, target_prob / draft_prob);
            float uniform_sample = uniform_samples[batch_idx * (T + 1) + t];
            if (uniform_sample >= alpha) {
                break;
            }
            output_tokens[batch_idx * (T + 1) + t] = draft_token;
            num_accepted_tokens++;
        }
    }
    __syncthreads();

    using BlockLoad = cub::BlockLoad<float, kBlockSize, kItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockReduceFloat = cub::BlockReduce<float, kBlockSize>;
    using BlockScan = cub::BlockScan<float, kBlockSize>;
    using BlockReduceInt = cub::BlockReduce<int, kBlockSize>;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockReduceFloat::TempStorage reduce_float;
        typename BlockScan::TempStorage scan;
        typename BlockReduceInt::TempStorage reduce_int;
    } temp_storage;

    if (num_accepted_tokens < T) {
        float draft_prob_items[kItemsPerThread];
        float target_prob_items[kItemsPerThread];
        BlockLoad(temp_storage.load)
            .Load(draft_probs + batch_idx * T * V + num_accepted_tokens * V, draft_prob_items, V,
                  0.0f);
        BlockLoad(temp_storage.load)
            .Load(target_probs + batch_idx * T * V + num_accepted_tokens * V, target_prob_items, V,
                  0.0f);
        float adjusted_prob_items[kItemsPerThread];
#pragma unroll
        for (int i = 0; i < kItemsPerThread; ++i) {
            adjusted_prob_items[i] = fmaxf(0.0f, target_prob_items[i] - draft_prob_items[i]);
        }
        float adjusted_sum = BlockReduceFloat(temp_storage.reduce_float).Sum(adjusted_prob_items);
        __shared__ float adjusted_sum_s;
        if (threadIdx.x == 0) {
            adjusted_sum_s = adjusted_sum;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < kItemsPerThread; ++i) {
            if (adjusted_sum_s > 0.0f) {
                adjusted_prob_items[i] /= adjusted_sum_s;
            } else {
                adjusted_prob_items[i] = 1.0f / V;
            }
        }
        BlockScan(temp_storage.scan).InclusiveSum(adjusted_prob_items, adjusted_prob_items);
        __syncthreads();
        int candidate_tokens[kItemsPerThread];
        float uniform_sample = uniform_samples[batch_idx * (T + 1) + T];
#pragma unroll
        for (int i = 0; i < kItemsPerThread; ++i) {
            if (adjusted_prob_items[i] >= uniform_sample) {
                candidate_tokens[i] = threadIdx.x * kItemsPerThread + i;
            } else {
                candidate_tokens[i] = V;
            }
        }
        int sampled_token = BlockReduceInt(temp_storage.reduce_int)
                                .Reduce(candidate_tokens, thrust::minimum<int>());
        if (threadIdx.x == 0) {
            output_tokens[batch_idx * (T + 1) + num_accepted_tokens] = min(sampled_token, V - 1);
        }
    } else {
        float target_prob_items[kItemsPerThread];
        BlockLoad(temp_storage.load)
            .Load(target_probs + batch_idx * T * V + (num_accepted_tokens - 1) * V,
                  target_prob_items, V, 0.0f);
        BlockScan(temp_storage.scan).InclusiveSum(target_prob_items, target_prob_items);
        __syncthreads();
        int candidate_tokens[kItemsPerThread];
        float uniform_sample = uniform_samples[batch_idx * (T + 1) + num_accepted_tokens];
#pragma unroll
        for (int i = 0; i < kItemsPerThread; i++) {
            if (target_prob_items[i] >= uniform_sample) {
                candidate_tokens[i] = threadIdx.x * kItemsPerThread + i;
            } else {
                candidate_tokens[i] = V;
            }
        }
        int sampled_token = BlockReduceInt(temp_storage.reduce_int)
                                .Reduce(candidate_tokens, thrust::minimum<int>());
        if (threadIdx.x == 0) {
            output_tokens[batch_idx * (T + 1) + num_accepted_tokens] = min(sampled_token, V - 1);
        }
    }
}

// draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens are device pointers
extern "C" void solve(const int* draft_tokens, const float* draft_probs, const float* target_probs,
                      const float* uniform_samples, int* output_tokens, int B, int T, int V) {
    constexpr int kBlockSize = 256;
    constexpr int kItemsPerThread = 128;
    verify_kernel<kBlockSize, kItemsPerThread><<<B, kBlockSize>>>(
        draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens, B, T, V);
    cudaDeviceSynchronize();
}
