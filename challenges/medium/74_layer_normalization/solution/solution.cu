#include <cub/cub.cuh>
#include <cuda_runtime.h>

struct ReduceOp {
    __device__ float2 operator()(float2 a, float2 b) const {
        return make_float2(a.x + b.x, a.y + b.y);
    }
};

template <int BlockSize, int ItemsPerThread>
__global__ void layer_norm_kernel(const float* input, const float* weight, const float* bias,
                                  float* output, int N, int C, float eps) {
    using BlockLoad = cub::BlockLoad<float, BlockSize, ItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockReduce = cub::BlockReduce<float2, BlockSize>;
    using BlockStore =
        cub::BlockStore<float, BlockSize, ItemsPerThread, cub::BLOCK_STORE_VECTORIZE>;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockReduce::TempStorage reduce;
        typename BlockStore::TempStorage store;
    } temp_storage;

    float input_items[ItemsPerThread];
    BlockLoad(temp_storage.load).Load(input + blockIdx.x * C, input_items, C, 0.0f);

    float sum = 0.0f;
    float square_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ItemsPerThread; i++) {
        sum += input_items[i];
        square_sum += input_items[i] * input_items[i];
    }
    float2 sum_sq_sum = make_float2(sum, square_sum);
    float2 block_sum_sq_sum = BlockReduce(temp_storage.reduce).Reduce(sum_sq_sum, ReduceOp());

    __shared__ float mean;
    __shared__ float inv_std;
    if (threadIdx.x == 0) {
        float block_sum = block_sum_sq_sum.x;
        float block_square_sum = block_sum_sq_sum.y;
        mean = block_sum / C;
        float variance = block_square_sum / C - mean * mean;
        inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();

    float weight_items[ItemsPerThread];
    float bias_items[ItemsPerThread];
    BlockLoad(temp_storage.load).Load(weight, weight_items, C, 0.0f);
    BlockLoad(temp_storage.load).Load(bias, bias_items, C, 0.0f);
#pragma unroll
    for (int i = 0; i < ItemsPerThread; i++) {
        input_items[i] = weight_items[i] * (input_items[i] - mean) * inv_std + bias_items[i];
    }

    BlockStore(temp_storage.store).Store(output + blockIdx.x * C, input_items, C);
}

// input, weight, bias, output are device pointers
extern "C" void solve(const float* input, const float* weight, const float* bias, float* output,
                      int N, int C, float eps) {
    constexpr int kBlockSize = 256;
    constexpr int kItemsPerThread = 4;
    layer_norm_kernel<kBlockSize, kItemsPerThread>
        <<<N, kBlockSize>>>(input, weight, bias, output, N, C, eps);
    cudaDeviceSynchronize();
}
