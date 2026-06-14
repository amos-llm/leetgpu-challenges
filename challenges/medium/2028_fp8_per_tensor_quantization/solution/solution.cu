#include <cfloat>
#include <cub/cub.cuh>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

template <int kBlockSize>
__global__ void find_max_abs_kernel(const float* x, float* max_abs, int N) {
    using BlockReduce = cub::BlockReduce<float, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int vec_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx = vec_idx * 4;
    float local_max_abs = 0;
    if (global_idx + 3 < N) {
        float4 x_vec = reinterpret_cast<const float4*>(x)[vec_idx];
        local_max_abs = fmaxf(local_max_abs, fabsf(x_vec.x));
        local_max_abs = fmaxf(local_max_abs, fabsf(x_vec.y));
        local_max_abs = fmaxf(local_max_abs, fabsf(x_vec.z));
        local_max_abs = fmaxf(local_max_abs, fabsf(x_vec.w));
    } else {
        for (int idx = global_idx; idx < N; ++idx) {
            local_max_abs = fmaxf(local_max_abs, fabsf(x[idx]));
        }
    }
    local_max_abs = BlockReduce(temp_storage).Reduce(local_max_abs, cuda::maximum<float>());
    if (threadIdx.x == 0) {
        int* max_abs_int = reinterpret_cast<int*>(max_abs);
        int expected = *max_abs_int;
        int desired;
        do {
            desired = __float_as_int(fmaxf(local_max_abs, __int_as_float(expected)));
            expected = atomicCAS(max_abs_int, expected, desired);
        } while (expected != desired);
    }
}

__global__ void fp8_per_tensor_quantize_kernel(const float* x, const float* max_abs, uint8_t* y,
                                               float* scale_out, int N) {
    int vec_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx = vec_idx * 4;
    float scale = *max_abs / 448.0f;
    if (scale == 0.0f) {
        scale = 1.0f;
    }
    if (global_idx + 3 < N) {
        float4 x_vec = reinterpret_cast<const float4*>(x)[vec_idx];
        x_vec.x /= scale;
        x_vec.y /= scale;
        x_vec.z /= scale;
        x_vec.w /= scale;
        __nv_fp8x4_e4m3 y_vec(x_vec);
        reinterpret_cast<uint32_t*>(y)[vec_idx] = y_vec.__x;
    } else {
        for (int idx = global_idx; idx < N; ++idx) {
            y[idx] = __nv_fp8_e4m3(x[idx] / scale).__x;
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *scale_out = scale;
    }
}

// x, y, scale_out are device pointers
extern "C" void solve(const float* x, uint8_t* y, float* scale_out, int N) {
    int num_vectors = (N + 3) / 4;
    constexpr int kBlockSize = 256;
    int blocks_per_grid = (num_vectors + kBlockSize - 1) / kBlockSize;
    float* d_max_abs = nullptr;
    cudaMalloc(&d_max_abs, sizeof(float));
    cudaMemset(d_max_abs, 0, sizeof(float));
    find_max_abs_kernel<kBlockSize><<<blocks_per_grid, kBlockSize>>>(x, d_max_abs, N);
    fp8_per_tensor_quantize_kernel<<<blocks_per_grid, kBlockSize>>>(x, d_max_abs, y, scale_out, N);
    cudaDeviceSynchronize();
    cudaFree(d_max_abs);
}
