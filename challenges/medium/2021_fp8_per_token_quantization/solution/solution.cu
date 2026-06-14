#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdint.h>

template <int kBlockSize>
__global__ void fp8_per_token_quantize_vectorized_kernel(const float* x, uint8_t* y, float* scales,
                                                         int K) {
    using BlockReduce = cub::BlockReduce<float, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float global_max_abs;

    int token_idx = blockIdx.x;
    const float* x_base = x + token_idx * K;
    uint8_t* y_base = y + token_idx * K;
    int num_vecs = K / 4;

    float local_max_abs = 0.0f;
    for (int vec_idx = threadIdx.x; vec_idx < num_vecs; vec_idx += blockDim.x) {
        float4 x_vec = reinterpret_cast<const float4*>(x_base)[vec_idx];
        local_max_abs = fmaxf(local_max_abs, fabsf(x_vec.x));
        local_max_abs = fmaxf(local_max_abs, fabsf(x_vec.y));
        local_max_abs = fmaxf(local_max_abs, fabsf(x_vec.z));
        local_max_abs = fmaxf(local_max_abs, fabsf(x_vec.w));
    }
    local_max_abs = BlockReduce(temp_storage).Reduce(local_max_abs, cuda::maximum<float>());
    if (threadIdx.x == 0) {
        global_max_abs = local_max_abs;
    }
    __syncthreads();

    float scale = global_max_abs / 448.0f;
    if (scale == 0.0f) {
        scale = 1.0f;
    }
    for (int vec_idx = threadIdx.x; vec_idx < num_vecs; vec_idx += blockDim.x) {
        float4 x_vec = reinterpret_cast<const float4*>(x_base)[vec_idx];
        x_vec.x /= scale;
        x_vec.y /= scale;
        x_vec.z /= scale;
        x_vec.w /= scale;
        reinterpret_cast<uint32_t*>(y_base)[vec_idx] = __nv_fp8x4_e4m3(x_vec).__x;
    }

    if (threadIdx.x == 0) {
        scales[token_idx] = scale;
    }
}

template <int kBlockSize>
__global__ void fp8_per_token_quantize_scalar_kernel(const float* x, uint8_t* y, float* scales,
                                                     int K) {
    using BlockReduce = cub::BlockReduce<float, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float global_max_abs;

    int token_idx = blockIdx.x;
    const float* x_base = x + token_idx * K;
    uint8_t* y_base = y + token_idx * K;

    float local_max_abs = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        local_max_abs = fmaxf(local_max_abs, fabsf(x_base[i]));
    }
    local_max_abs = BlockReduce(temp_storage).Reduce(local_max_abs, cuda::maximum<float>());
    if (threadIdx.x == 0) {
        global_max_abs = local_max_abs;
    }
    __syncthreads();

    float scale = global_max_abs / 448.0f;
    if (scale == 0.0f) {
        scale = 1.0f;
    }
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float val = x_base[i];
        y_base[i] = __nv_fp8_e4m3(val / scale).__x;
    }

    if (threadIdx.x == 0) {
        scales[token_idx] = scale;
    }
}

// x, y, scales are device pointers
extern "C" void solve(const float* x, uint8_t* y, float* scales, int M, int K) {
    constexpr int kBlockSize = 256;
    constexpr int kVecAlign = alignof(float4);
    bool x_aligned = reinterpret_cast<uintptr_t>(x) % kVecAlign == 0;
    bool y_aligned = reinterpret_cast<uintptr_t>(y) % kVecAlign == 0;
    if (K % 4 == 0 && x_aligned && y_aligned) {
        fp8_per_token_quantize_vectorized_kernel<kBlockSize><<<M, kBlockSize>>>(x, y, scales, K);
    } else {
        fp8_per_token_quantize_scalar_kernel<kBlockSize><<<M, kBlockSize>>>(x, y, scales, K);
    }
    cudaDeviceSynchronize();
}
