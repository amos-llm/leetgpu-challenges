#include <cstdio>
#include <cuda_runtime.h>
#include <float.h>

template <int kK>
__device__ void insert_topk(float* weights, int* indices, float weight, int index) {
#pragma unroll
    for (int i = 0; i < kK; ++i) {
        if (weight > weights[i]) {
#pragma unroll
            for (int j = kK - 1; j > i; --j) {
                weights[j] = weights[j - 1];
                indices[j] = indices[j - 1];
            }
            weights[i] = weight;
            indices[i] = index;
            return;
        }
    }
}

template <int kK>
__device__ void merge_topk(float* dst_weights, int* dst_indices, const float* src_weights,
                           const int* src_indices) {
#pragma unroll
    for (int i = 0; i < kK; ++i) {
        insert_topk<kK>(dst_weights, dst_indices, src_weights[i], src_indices[i]);
    }
}

template <int kK>
__global__ void moe_topk_kernel(const float* __restrict__ logits, // [M, E]
                                float* __restrict__ topk_weights, // [M, kK]
                                int* __restrict__ topk_indices,   // [M, kK]
                                int M, int E) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int rows_per_block = blockDim.x / 32;
    int row_idx = blockIdx.x * rows_per_block + warp_id;
    if (row_idx >= M) {
        return;
    }

    float top_weights[kK];
    int top_indices[kK];
#pragma unroll
    for (int i = 0; i < kK; ++i) {
        top_weights[i] = -FLT_MAX;
        top_indices[i] = -1;
    }

    for (int e = lane_id; e < E; e += 32) {
        float logit = logits[row_idx * E + e];
        insert_topk<kK>(top_weights, top_indices, logit, e);
    }

#pragma unroll
    for (int s = 16; s > 0; s /= 2) {
        float other_weights[kK];
        int other_indices[kK];
#pragma unroll
        for (int i = 0; i < kK; ++i) {
            other_weights[i] = __shfl_down_sync(0xffffffff, top_weights[i], s);

            other_indices[i] = __shfl_down_sync(0xffffffff, top_indices[i], s);
        }
        if (lane_id < s) {
            merge_topk<kK>(top_weights, top_indices, other_weights, other_indices);
        }
    }

    if (lane_id == 0) {
        float max_weight = top_weights[0];
        float weight_exp_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < kK; ++i) {
            top_weights[i] = expf(top_weights[i] - max_weight);
            weight_exp_sum += top_weights[i];
        }
#pragma unroll
        for (int i = 0; i < kK; ++i) {
            topk_weights[row_idx * kK + i] = top_weights[i] / weight_exp_sum;
            topk_indices[row_idx * kK + i] = top_indices[i];
        }
    }
}

extern "C" void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k) {
    constexpr int kBlockSize = 256;
    constexpr int kRowsPerBlock = kBlockSize / 32;
    int num_blocks = (M + kRowsPerBlock - 1) / kRowsPerBlock;
#define DISPATCH_KERNEL(CASE_K)                                                                    \
    case CASE_K:                                                                                   \
        moe_topk_kernel<CASE_K>                                                                    \
            <<<num_blocks, kBlockSize>>>(logits, topk_weights, topk_indices, M, E);                \
        break;
    switch (k) {
        DISPATCH_KERNEL(1)
        DISPATCH_KERNEL(2)
        DISPATCH_KERNEL(3)
        DISPATCH_KERNEL(4)
        DISPATCH_KERNEL(5)
        DISPATCH_KERNEL(6)
        DISPATCH_KERNEL(7)
        DISPATCH_KERNEL(8)
    default:
        fprintf(stderr, "Unsupported K: %d\n", k);
    }
#undef DISPATCH_KERNEL
    cudaDeviceSynchronize();
}
