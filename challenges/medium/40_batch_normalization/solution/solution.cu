#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int BlockSize>
__global__ void batch_norm_kernel(const float* input, const float* gamma, const float* beta,
                                  float* output, int N, int C, float eps) {
    using BlockReduce = cub::BlockReduce<float, BlockSize>;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float x = input[i * C + blockIdx.x];
        sum += x;
        square_sum += x * x;
    }

    sum = BlockReduce(temp_storage).Sum(sum);
    __syncthreads();
    square_sum = BlockReduce(temp_storage).Sum(square_sum);

    __shared__ float mean;
    __shared__ float inv_std;
    if (threadIdx.x == 0) {
        mean = sum / N;
        float variance = square_sum / N - mean * mean;
        inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float x = input[i * C + blockIdx.x];
        float y = (x - mean) * inv_std * gamma[blockIdx.x] + beta[blockIdx.x];
        output[i * C + blockIdx.x] = y;
    }
}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output,
                      int N, int C, float eps) {
    constexpr int kBlockSize = 256;
    batch_norm_kernel<kBlockSize><<<C, kBlockSize>>>(input, gamma, beta, output, N, C, eps);
    cudaDeviceSynchronize();
}
