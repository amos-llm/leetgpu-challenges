#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdint.h>

template <int kBlockSize>
__global__ void fp8_2d_block_quantize_kernel(const float* x, uint8_t* y, float* scales, int M,
                                             int N, int BLOCK_SIZE) {
    using BlockReduce = cub::BlockReduce<float, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float group_max_abs;

    if (threadIdx.x == 0) {
        group_max_abs = 0.0f;
    }
    __syncthreads();

    int tile_m = blockIdx.x;
    int tile_n = blockIdx.y;
    int m_start = tile_m * BLOCK_SIZE;
    int m_end = min(m_start + BLOCK_SIZE, M);
    int n_start = tile_n * BLOCK_SIZE;
    int n_end = min(n_start + BLOCK_SIZE, N);
    for (int m = m_start; m < m_end; ++m) {
        int idx = threadIdx.x + n_start;
        float val = idx < n_end ? fabsf(x[m * N + idx]) : 0.0f;
        val = BlockReduce(temp_storage).Reduce(val, cuda::maximum<float>());
        if (threadIdx.x == 0) {
            group_max_abs = fmaxf(group_max_abs, val);
        }
        __syncthreads();
    }

    float scale = group_max_abs / 448.0f;
    if (scale == 0.0f) {
        scale = 1.0f;
    }

    for (int m = m_start; m < m_end; ++m) {
        int idx = threadIdx.x + n_start;
        if (idx < n_end) {
            y[m * N + idx] = __nv_fp8_e4m3(x[m * N + idx] / scale).__x;
        }
    }
    if (threadIdx.x == 0) {
        scales[tile_m * gridDim.y + tile_n] = scale;
    }
}

// x, y, scales are device pointers
extern "C" void solve(const float* x, uint8_t* y, float* scales, int M, int N, int BLOCK_SIZE) {
    int num_blocks_m = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_n = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid_dim(num_blocks_m, num_blocks_n);
    constexpr int kBlockSize = 128;
    fp8_2d_block_quantize_kernel<kBlockSize>
        <<<grid_dim, kBlockSize>>>(x, y, scales, M, N, BLOCK_SIZE);
    cudaDeviceSynchronize();
}
