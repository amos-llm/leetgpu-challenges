#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int kBlockSize, int kItemsPerThread>
__global__ void rms_norm_kernel(const float* x, float* residual, const float* weight, float* out,
                                int N, int C, float eps) {
    using BlockLoad = cub::BlockLoad<float, kBlockSize, kItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockReduce = cub::BlockReduce<float, kBlockSize>;
    using BlockStore =
        cub::BlockStore<float, kBlockSize, kItemsPerThread, cub::BLOCK_STORE_VECTORIZE>;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockReduce::TempStorage reduce;
        typename BlockStore::TempStorage store;
    } temp_storage;

    float x_items[kItemsPerThread];
    float r_items[kItemsPerThread];
    BlockLoad(temp_storage.load).Load(x + blockIdx.x * C, x_items, C, 0.0f);
    BlockLoad(temp_storage.load).Load(residual + blockIdx.x * C, r_items, C, 0.0f);

    float z_items[kItemsPerThread];
    float z_square_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
        float z = x_items[i] + r_items[i];
        z_items[i] = z;
        z_square_sum += z * z;
    }
    __shared__ float rrms;
    float sum = BlockReduce(temp_storage.reduce).Sum(z_square_sum);
    if (threadIdx.x == 0) {
        rrms = rsqrtf(sum / C + eps);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
        if (threadIdx.x * kItemsPerThread + i < C) {
            z_items[i] *= rrms * weight[threadIdx.x * kItemsPerThread + i];
        }
    }
    BlockStore(temp_storage.store).Store(out + blockIdx.x * C, z_items, C);
}

// x, residual, weight, out are device pointers
extern "C" void solve(const float* x, float* residual, const float* weight, float* out, int N,
                      int C, float eps) {
    constexpr int kBlockSize = 1024;
    constexpr int kItemsPerThread = 4;
    constexpr int kItemsPerBlock = kBlockSize * kItemsPerThread;
    rms_norm_kernel<kBlockSize, kItemsPerThread>
        <<<N, kBlockSize>>>(x, residual, weight, out, N, C, eps);
    cudaDeviceSynchronize();
}
