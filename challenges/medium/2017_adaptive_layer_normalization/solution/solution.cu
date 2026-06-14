#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int kBlockSize>
__global__ void adaptive_layer_norm_kernel(const float* X, const float* scale, const float* shift,
                                           float* output, int B, int N, int D) {
    using BlockReduce = cub::BlockReduce<float, kBlockSize>;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += kBlockSize) {
        float x = X[blockIdx.x * D + i];
        sum += x;
        square_sum += x * x;
    }

    __shared__ float mean;
    __shared__ float inv_std;
    sum = BlockReduce(temp_storage).Sum(sum);
    __syncthreads();
    square_sum = BlockReduce(temp_storage).Sum(square_sum);
    if (threadIdx.x == 0) {
        mean = sum / D;
        inv_std = rsqrtf(square_sum / D - mean * mean + 1e-5f);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < D; i += kBlockSize) {
        float x = (X[blockIdx.x * D + i] - mean) * inv_std;
        float y = x * (1 + scale[blockIdx.x / N * D + i]) + shift[blockIdx.x / N * D + i];
        output[blockIdx.x * D + i] = y;
    }
}

// X, scale, shift, output are device pointers
extern "C" void solve(const float* X, const float* scale, const float* shift, float* output, int B,
                      int N, int D) {
    constexpr int kBlockSize = 256;
    adaptive_layer_norm_kernel<kBlockSize><<<B * N, kBlockSize>>>(X, scale, shift, output, B, N, D);
    cudaDeviceSynchronize();
}
