#include <cub/cub.cuh>
#include <cuda_runtime.h>

struct ReduceOp {
    __device__ float2 operator()(float2 a, float2 b) const {
        return make_float2(a.x + b.x, a.y + b.y);
    }
};

template <int BlockSize>
__global__ void batch_norm_kernel(const float* input, const float* gamma, const float* beta,
                                  float* output, int N, int C, float eps) {
    using BlockReduce = cub::BlockReduce<float2, BlockSize>;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float x = input[i * C + blockIdx.x];
        sum += x;
        square_sum += x * x;
    }

    float2 sum_sq_sum = make_float2(sum, square_sum);
    float2 block_sum_sq_sum = BlockReduce(temp_storage).Reduce(sum_sq_sum, ReduceOp());

    __shared__ float mean;
    __shared__ float inv_std;
    if (threadIdx.x == 0) {
        float sum = block_sum_sq_sum.x;
        float square_sum = block_sum_sq_sum.y;
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
