#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int BlockSize> __global__ void softmax_kernel(const float* input, float* output, int N) {
    struct MaxSum {
        float max;
        float sum;
    };

    struct ReduceOp {
        __device__ MaxSum operator()(MaxSum a, MaxSum b) const {
            float max = fmaxf(a.max, b.max);
            return MaxSum{max, a.sum * expf(a.max - max) + b.sum * expf(b.max - max)};
        }
    };

    using BlockReduce = cub::BlockReduce<MaxSum, BlockSize>;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    MaxSum max_sum = {-1e38f, 0.0f};

    for (int i = 0; i < N; i += BlockSize) {
        int j = i + threadIdx.x;
        if (j < N) {
            float value = input[j];
            float max = fmaxf(max_sum.max, value);
            max_sum.sum = max_sum.sum * expf(max_sum.max - max) + expf(value - max);
            max_sum.max = max;
        }
    }

    MaxSum agg_max_sum = BlockReduce(temp_storage).Reduce(max_sum, ReduceOp());
    __shared__ float row_max;
    __shared__ float row_sum;
    if (threadIdx.x == 0) {
        row_max = agg_max_sum.max;
        row_sum = agg_max_sum.sum;
    }
    __syncthreads();

    for (int i = 0; i < N; i += BlockSize) {
        int j = i + threadIdx.x;
        if (j < N) {
            output[j] = expf(input[j] - row_max) / row_sum;
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    constexpr int kBlockSize = 1024;
    softmax_kernel<kBlockSize><<<1, kBlockSize>>>(input, output, N);
    cudaDeviceSynchronize();
}
