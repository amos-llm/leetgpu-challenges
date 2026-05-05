#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/functional.h>

template <int BlockSize>
__global__ void local_scan_kernel(const float* values, const int* flags, float* prefix_sums,
                                  int* first_flag_indices, float* last_segment_sums, int N) {
    struct FlagValue {
        int flag;
        float value;
    };

    struct ScanOp {
        __device__ FlagValue operator()(FlagValue a, FlagValue b) const {
            if (b.flag == 1) {
                return FlagValue{b.flag, b.value};
            }
            return FlagValue{a.flag, a.value + b.value};
        }
    };

    using BlockReduce = cub::BlockReduce<int, BlockSize>;
    using BlockScan = cub::BlockScan<FlagValue, BlockSize>;

    __shared__ union {
        typename BlockReduce::TempStorage reduce;
        typename BlockScan::TempStorage scan;
    } temp_storage;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int flag = 0;
    float value = 0.0f;
    if (idx < N) {
        flag = flags[idx];
        value = values[idx];
    }

    int first_flag_index = BlockSize;
    if (idx < N && flag == 1) {
        first_flag_index = threadIdx.x;
    }
    int agg_first_flag_index =
        BlockReduce(temp_storage.reduce).Reduce(first_flag_index, thrust::minimum<int>());
    if (threadIdx.x == 0) {
        first_flag_indices[blockIdx.x] = agg_first_flag_index;
    }
    __syncthreads();

    FlagValue flag_value{flag, value};
    FlagValue agg_flag_value;
    BlockScan(temp_storage.scan).InclusiveScan(flag_value, flag_value, ScanOp(), agg_flag_value);
    if (idx < N) {
        prefix_sums[idx] = flag_value.value;
    }
    if (threadIdx.x == 0) {
        last_segment_sums[blockIdx.x] = agg_flag_value.value;
    }
}

template <int BlockSize>
__global__ void global_scan_kernel(const int* first_flag_indices, const float* last_segment_sums,
                                   float* offsets, int N) {
    struct IndexSum {
        int index;
        float sum;
    };

    struct ScanOp {
        __device__ IndexSum operator()(IndexSum a, IndexSum b) const {
            if (b.index < BlockSize) {
                return IndexSum{b.index, b.sum};
            }
            return IndexSum{a.index, a.sum + b.sum};
        }
    };

    using BlockScan = cub::BlockScan<IndexSum, BlockSize>;

    __shared__ typename BlockScan::TempStorage temp_storage;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = BlockSize;
    float sum = 0.0f;
    if (idx < N) {
        index = first_flag_indices[idx];
        sum = last_segment_sums[idx];
    }

    IndexSum index_sum{index, sum};
    BlockScan(temp_storage).InclusiveScan(index_sum, index_sum, ScanOp());
    if (idx < N) {
        offsets[idx] = index_sum.sum;
    }
}

void global_scan(const int* first_segment_indices, const float* last_segment_sums, float* offsets,
                 int N) {
    constexpr int kBlockSize = 256;
    int grid_size = (N + kBlockSize - 1) / kBlockSize;
    global_scan_kernel<kBlockSize>
        <<<grid_size, kBlockSize>>>(first_segment_indices, last_segment_sums, offsets, N);
}

template <int BlockSize>
__global__ void add_offset_kernel(const int* flags, const float* prefix_sums,
                                  const int* first_flag_indices, const float* offsets,
                                  float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    float value = 0.0f;
    if (threadIdx.x > 0 && flags[idx] != 1) {
        value = prefix_sums[idx - 1];
    }

    if (blockIdx.x > 0) {
        int first_flag_index = first_flag_indices[blockIdx.x];
        if (threadIdx.x < first_flag_index) {
            value += offsets[blockIdx.x - 1];
        }
    }

    output[idx] = value;
}

// values, flags, output are device pointers
extern "C" void solve(const float* values, const int* flags, float* output, int N) {
    constexpr int kBlockSize = 256;
    int grid_size = (N + kBlockSize - 1) / kBlockSize;
    float* d_prefix_sums;
    int* d_first_segment_indices;
    float* d_last_segment_sums;
    float* d_offsets;
    cudaMalloc(&d_prefix_sums, N * sizeof(float));
    cudaMalloc(&d_first_segment_indices, grid_size * sizeof(int));
    cudaMalloc(&d_last_segment_sums, grid_size * sizeof(float));
    cudaMalloc(&d_offsets, grid_size * sizeof(float));
    local_scan_kernel<kBlockSize><<<grid_size, kBlockSize>>>(
        values, flags, d_prefix_sums, d_first_segment_indices, d_last_segment_sums, N);
    global_scan(d_first_segment_indices, d_last_segment_sums, d_offsets, grid_size);
    add_offset_kernel<kBlockSize><<<grid_size, kBlockSize>>>(
        flags, d_prefix_sums, d_first_segment_indices, d_offsets, output, N);
    cudaDeviceSynchronize();
    cudaFree(d_offsets);
    cudaFree(d_last_segment_sums);
    cudaFree(d_first_segment_indices);
    cudaFree(d_prefix_sums);
}
