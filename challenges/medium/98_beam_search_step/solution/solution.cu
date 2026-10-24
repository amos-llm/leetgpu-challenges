#include <cuda_runtime.h>
#include <float.h>

template <int kK>
__device__ void insert_topk(float* scores, int* tokens, int* parents, float score, int token,
                            int parent) {
#pragma unroll
    for (int i = 0; i < kK; ++i) {
        if (score > scores[i]) {
#pragma unroll
            for (int j = kK - 1; j > i; --j) {
                scores[j] = scores[j - 1];
                tokens[j] = tokens[j - 1];
                parents[j] = parents[j - 1];
            }
            scores[i] = score;
            tokens[i] = token;
            parents[i] = parent;
            return;
        }
    }
}

template <int kK>
__device__ void merge_topk(float* dst_scores, int* dst_tokens, int* dst_parents,
                           const float* src_scores, const int* src_tokens, const int* src_parents) {
#pragma unroll
    for (int i = 0; i < kK; ++i) {
        insert_topk<kK>(dst_scores, dst_tokens, dst_parents, src_scores[i], src_tokens[i],
                        src_parents[i]);
    }
}

template <int kK> __device__ void reduce_topk(float* scores, int* tokens, int* parents) {
    int lane_id = threadIdx.x % 32;
#pragma unroll
    for (int s = 16; s > 0; s /= 2) {
        float other_scores[kK];
        int other_tokens[kK];
        int other_parents[kK];
#pragma unroll
        for (int i = 0; i < kK; ++i) {
            other_scores[i] = __shfl_down_sync(0xffffffff, scores[i], s);
            other_tokens[i] = __shfl_down_sync(0xffffffff, tokens[i], s);
            other_parents[i] = __shfl_down_sync(0xffffffff, parents[i], s);
        }
        if (lane_id < s) {
            merge_topk<kK>(scores, tokens, parents, other_scores, other_tokens, other_parents);
        }
    }
}

template <int kK>
__global__ void beam_search_step_kernel(const float* __restrict__ beam_scores,    // [B, kK]
                                        const float* __restrict__ token_logprobs, // [B, kK, V]
                                        float* __restrict__ new_beam_scores,      // [B, kK]
                                        int* __restrict__ parent_beam_indices,    // [B, kK]
                                        int* __restrict__ next_tokens,            // [B, kK]
                                        int V) {
    int batch_idx = blockIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    float top_scores[kK];
    int top_tokens[kK];
    int top_parents[kK];
#pragma unroll
    for (int i = 0; i < kK; ++i) {
        top_scores[i] = -FLT_MAX;
        top_tokens[i] = -1;
        top_parents[i] = -1;
    }

    int total_candidates = kK * V;
    for (int i = threadIdx.x; i < total_candidates; i += blockDim.x) {
        int beam_idx = i / V;
        int token_idx = i % V;
        float score = beam_scores[batch_idx * kK + beam_idx] +
                      token_logprobs[batch_idx * kK * V + beam_idx * V + token_idx];
        insert_topk<kK>(top_scores, top_tokens, top_parents, score, token_idx, beam_idx);
    }

    reduce_topk<kK>(top_scores, top_tokens, top_parents);

    __shared__ float shared_scores[32 * kK];
    __shared__ int shared_tokens[32 * kK];
    __shared__ int shared_parents[32 * kK];
    if (lane_id == 0) {
        for (int i = 0; i < kK; ++i) {
            shared_scores[warp_id * kK + i] = top_scores[i];
            shared_tokens[warp_id * kK + i] = top_tokens[i];
            shared_parents[warp_id * kK + i] = top_parents[i];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        float final_scores[kK];
        int final_tokens[kK];
        int final_parents[kK];
#pragma unroll
        for (int i = 0; i < kK; ++i) {
            final_scores[i] = shared_scores[lane_id * kK + i];
            final_tokens[i] = shared_tokens[lane_id * kK + i];
            final_parents[i] = shared_parents[lane_id * kK + i];
        }

        reduce_topk<kK>(final_scores, final_tokens, final_parents);

        if (lane_id == 0) {
#pragma unroll
            for (int i = 0; i < kK; ++i) {
                new_beam_scores[batch_idx * kK + i] = final_scores[i];
                next_tokens[batch_idx * kK + i] = final_tokens[i];
                parent_beam_indices[batch_idx * kK + i] = final_parents[i];
            }
        }
    }
}

extern "C" void solve(const float* beam_scores, const float* token_logprobs, float* new_beam_scores,
                      int* parent_beam_indices, int* next_tokens, int B, int K, int V) {
    constexpr int kBlockSize = 1024;
#define DISPATCH_KERNEL(CASE_K)                                                                    \
    case CASE_K:                                                                                   \
        beam_search_step_kernel<CASE_K><<<B, kBlockSize>>>(                                        \
            beam_scores, token_logprobs, new_beam_scores, parent_beam_indices, next_tokens, V);    \
        break;
    switch (K) {
        DISPATCH_KERNEL(1)
        DISPATCH_KERNEL(2)
        DISPATCH_KERNEL(3)
        DISPATCH_KERNEL(4)
        DISPATCH_KERNEL(5)
        DISPATCH_KERNEL(6)
        DISPATCH_KERNEL(7)
        DISPATCH_KERNEL(8)
    }
#undef DISPATCH_KERNEL
    cudaDeviceSynchronize();
}
