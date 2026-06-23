#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int kThreadsPerBlock>
__global__ void token_embedding_kernel(const int* token_ids, const int* position_ids,
                                       const float* token_embeddings,
                                       const float* position_embeddings, const float* gamma,
                                       const float* beta, float* output, int T, int D, float eps) {
    using BlockReduce = cub::BlockReduce<float, kThreadsPerBlock>;
    __shared__ typename BlockReduce::TempStorage temp_storage[2];

    int b = blockIdx.x / T;
    int t = blockIdx.x % T;
    int token_id = token_ids[b * T + t];
    int position_id = position_ids[t];

    float local_sum = 0.0f;
    float local_sum_square = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float s = token_embeddings[token_id * D + i] + position_embeddings[position_id * D + i];
        local_sum += s;
        local_sum_square += s * s;
    }
    float block_sum = BlockReduce(temp_storage[0]).Sum(local_sum);
    float block_sum_square = BlockReduce(temp_storage[1]).Sum(local_sum_square);

    __shared__ float mean;
    __shared__ float inv_std;
    if (threadIdx.x == 0) {
        mean = block_sum / D;
        float variance = block_sum_square / D - mean * mean;
        inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float s = token_embeddings[token_id * D + i] + position_embeddings[position_id * D + i];
        output[b * T * D + t * D + i] = gamma[i] * (s - mean) * inv_std + beta[i];
    }
}

// token_ids, position_ids, token_embeddings, position_embeddings, gamma, beta, output are device
// pointers
extern "C" void solve(const int* token_ids, const int* position_ids, const float* token_embeddings,
                      const float* position_embeddings, const float* gamma, const float* beta,
                      float* output, int B, int T, int V, int P, int D, float eps) {
    constexpr int kThreadsPerBlock = 256;
    token_embedding_kernel<kThreadsPerBlock>
        <<<B * T, kThreadsPerBlock>>>(token_ids, position_ids, token_embeddings,
                                      position_embeddings, gamma, beta, output, T, D, eps);
    cudaDeviceSynchronize();
}
