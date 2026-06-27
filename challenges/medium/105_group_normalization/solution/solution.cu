#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int kThreadsPerBlock>
__global__ void group_norm_kernel(const float* X, const float* gamma, const float* beta, float* Y,
                                  int N, int C, int H, int W, int G, float eps) {
    using BlockReduce = cub::BlockReduce<float, kThreadsPerBlock>;
    __shared__ typename BlockReduce::TempStorage temp_storage[2];

    int group_idx = blockIdx.x;
    int S = (C / G) * H * W;

    const float* X_base = X + group_idx * S;
    float* Y_base = Y + group_idx * S;

    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int i = threadIdx.x; i < S; i += blockDim.x) {
        sum += X_base[i];
        square_sum += X_base[i] * X_base[i];
    }
    float block_sum;
    float block_square_sum;
    block_sum = BlockReduce(temp_storage[0]).Sum(sum);
    block_square_sum = BlockReduce(temp_storage[1]).Sum(square_sum);
    __shared__ float mean;
    __shared__ float inv_std;
    if (threadIdx.x == 0) {
        mean = block_sum / S;
        float variance = block_square_sum / S - mean * mean;
        inv_std = rsqrt(variance + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < S; i += blockDim.x) {
        int channel_idx = (blockIdx.x % G) * (C / G) + (i / (H * W));
        Y_base[i] = gamma[channel_idx] * (X_base[i] - mean) * inv_std + beta[channel_idx];
    }
}

// X, gamma, beta, Y are device pointers
extern "C" void solve(const float* X, const float* gamma, const float* beta, float* Y, int N, int C,
                      int H, int W, int G, float eps) {
    constexpr int kThreadsPerBlock = 256;
    int num_groups = N * G;
    group_norm_kernel<kThreadsPerBlock>
        <<<num_groups, kThreadsPerBlock>>>(X, gamma, beta, Y, N, C, H, W, G, eps);
    cudaDeviceSynchronize();
}
