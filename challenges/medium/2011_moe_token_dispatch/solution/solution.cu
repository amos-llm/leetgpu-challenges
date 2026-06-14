#include <cub/cub.cuh>
#include <cuda_runtime.h>

__global__ void count_tokens_kernel(const int* expert_idx, // [T]
                                    int* token_counts,     // [E]
                                    int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < T) {
        int e = expert_idx[idx];
        atomicAdd(token_counts + e, 1);
    }
}

__global__ void compute_offsets_kernel(const int* token_counts,   // [E]
                                       int* expert_start_offsets, // [E]
                                       int E) {
    int prefix_sum = 0;
    for (int e = 0; e < E; ++e) {
        expert_start_offsets[e] = prefix_sum;
        prefix_sum += token_counts[e];
    }
}

__global__ void init_identity_kernel(int* identity, // [T]
                                     int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < T) {
        identity[idx] = idx;
    }
}

__global__ void dispatch_kernel(const float* x,                  // [T, D]
                                const int* sorted_expert_idx,    // [T]
                                const int* sorted_token_idx,     // [T]
                                const int* expert_start_offsets, // [E]
                                float* dispatched_x,             // [E, capacity, D]
                                int T, int D, int capacity) {
    int idx = blockIdx.x;
    if (idx >= T) {
        return;
    }

    int expert_idx = sorted_expert_idx[idx];
    int token_idx = sorted_token_idx[idx];

    int start_offset = expert_start_offsets[expert_idx];
    int slot = idx - start_offset;

    const float* in = x + token_idx * D;
    float* out = dispatched_x + expert_idx * capacity * D + slot * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        out[d] = in[d];
    }
}

extern "C" void solve(const float* x,        // [T, D]
                      const int* expert_idx, // [T]
                      float* dispatched_x,   // [E, capacity, D]
                      int* token_counts,     // [E]
                      int T, int D, int E, int capacity) {
    count_tokens_kernel<<<(T + 255) / 256, 256>>>(expert_idx, token_counts, T);

    int* d_expert_start_offsets;
    cudaMalloc(&d_expert_start_offsets, E * sizeof(int));
    compute_offsets_kernel<<<1, 1>>>(token_counts, d_expert_start_offsets, E);

    int* d_identity;
    cudaMalloc(&d_identity, T * sizeof(int));
    init_identity_kernel<<<(T + 255) / 256, 256>>>(d_identity, T);

    int* d_sorted_expert_idx;
    int* d_sorted_token_idx;
    cudaMalloc(&d_sorted_expert_idx, T * sizeof(int));
    cudaMalloc(&d_sorted_token_idx, T * sizeof(int));

    void* d_temp_storage = nullptr;
    size_t d_temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, d_temp_storage_bytes, expert_idx,
                                    d_sorted_expert_idx, d_identity, d_sorted_token_idx, T);
    cudaMalloc(&d_temp_storage, d_temp_storage_bytes);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, d_temp_storage_bytes, expert_idx,
                                    d_sorted_expert_idx, d_identity, d_sorted_token_idx, T);

    dispatch_kernel<<<T, 256>>>(x, d_sorted_expert_idx, d_sorted_token_idx, d_expert_start_offsets,
                                dispatched_x, T, D, capacity);

    cudaDeviceSynchronize();
    cudaFree(d_temp_storage);
    cudaFree(d_sorted_token_idx);
    cudaFree(d_sorted_expert_idx);
    cudaFree(d_identity);
    cudaFree(d_expert_start_offsets);
}
