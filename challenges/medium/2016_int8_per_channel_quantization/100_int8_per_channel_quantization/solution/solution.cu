#include <algorithm>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdint.h>

template <int kBlockSize>
__global__ void int8_per_channel_quantize_kernel(const float* x, int8_t* y, float* scales, int M,
                                                 int K) {
    using BlockReduce = cub::BlockReduce<float, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float global_max_abs;

    int channel_idx = blockIdx.x;

    float local_max_abs = 0.0f;
    for (int token_idx = threadIdx.x; token_idx < M; token_idx += blockDim.x) {
        local_max_abs = fmaxf(local_max_abs, fabsf(x[token_idx * K + channel_idx]));
    }
    local_max_abs = BlockReduce(temp_storage).Reduce(local_max_abs, cuda::maximum<float>());
    if (threadIdx.x == 0) {
        global_max_abs = local_max_abs;
    }
    __syncthreads();

    float scale = global_max_abs / 127.0f;
    if (scale == 0.0f) {
        scale = 1.0f;
    }

    for (int token_idx = threadIdx.x; token_idx < M; token_idx += blockDim.x) {
        float val = x[token_idx * K + channel_idx];
        y[token_idx * K + channel_idx] =
            __float2int_rn(cuda::std::clamp(val / scale, -128.0f, 127.0f));
    }

    if (threadIdx.x == 0) {
        scales[channel_idx] = scale;
    }
}

// x, y, scales are device pointers
extern "C" void solve(const float* x, int8_t* y, float* scales, int M, int K) {
    constexpr int kBlockSize = 256;
    int8_per_channel_quantize_kernel<kBlockSize><<<K, kBlockSize>>>(x, y, scales, M, K);
    cudaDeviceSynchronize();
}
