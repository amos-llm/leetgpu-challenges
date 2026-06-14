#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdint.h>

template <int kBlockSize>
__global__ void fp8_1d_block_quantize_kernel(const float* x, uint8_t* y, float* scales, int K,
                                             int BLOCK_SIZE) {
    using BlockReduce = cub::BlockReduce<float, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float group_max_abs;

    int token_idx = blockIdx.x;
    int num_groups = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int gid = 0; gid < num_groups; ++gid) {
        int k_start = gid * BLOCK_SIZE;
        int k_end = min(k_start + BLOCK_SIZE, K);
        int idx = k_start + threadIdx.x;
        float val = idx < k_end ? fabsf(x[token_idx * K + idx]) : 0.0f;
        val = BlockReduce(temp_storage).Reduce(val, cuda::maximum<float>());
        if (threadIdx.x == 0) {
            group_max_abs = val;
        }
        __syncthreads();
        float scale = group_max_abs / 448.0f;
        if (scale == 0.0f) {
            scale = 1.0f;
        }
        if (idx < K) {
            y[token_idx * K + idx] = __nv_fp8_e4m3(x[token_idx * K + idx] / scale).__x;
        }
        if (threadIdx.x == 0) {
            scales[token_idx * num_groups + gid] = scale;
        }
        __syncthreads();
    }
}

// x, y, scales are device pointers
extern "C" void solve(const float* x, uint8_t* y, float* scales, int M, int K, int BLOCK_SIZE) {
    constexpr int kBlockSize = 128;
    fp8_1d_block_quantize_kernel<kBlockSize><<<M, kBlockSize>>>(x, y, scales, K, BLOCK_SIZE);
    cudaDeviceSynchronize();
}
