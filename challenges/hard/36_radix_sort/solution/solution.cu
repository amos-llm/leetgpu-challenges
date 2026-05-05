#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <unsigned int NumBins>
__global__ void histogram_kernel(const unsigned int* input, unsigned int* block_histograms, int N,
                                 unsigned int shift_bits) {
    __shared__ unsigned int s_histograms[NumBins];

    for (int i = threadIdx.x; i < NumBins; i += blockDim.x) {
        s_histograms[i] = 0;
    }
    __syncthreads();

    int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (input_idx < N) {
        unsigned int bin = (input[input_idx] >> shift_bits) & (NumBins - 1);
        atomicAdd(&s_histograms[bin], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < NumBins; i += blockDim.x) {
        block_histograms[i * gridDim.x + blockIdx.x] = s_histograms[i];
    }
}

template <int BlockSize, unsigned int NumBins>
__global__ void scatter_kernel(const unsigned int* input, const unsigned int* block_bin_offsets,
                               unsigned int* output, int N, unsigned int shift_bits) {
    using BlockScan = cub::BlockScan<int, BlockSize>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int input_idx = blockIdx.x * BlockSize + threadIdx.x;
    if (input_idx < N) {
        unsigned int bin = (input[input_idx] >> shift_bits) & (NumBins - 1);
        for (int i = 0; i < NumBins; ++i) {
            int is_in_bin = 0;
            if (i == bin) {
                is_in_bin = 1;
            }
            int local_idx;
            BlockScan(temp_storage).ExclusiveSum(is_in_bin, local_idx);
            __syncthreads();
            if (i == bin) {
                int output_idx = block_bin_offsets[i * gridDim.x + blockIdx.x] + local_idx;
                output[output_idx] = input[input_idx];
            }
        }
    }
}

// input, output are device pointers
extern "C" void solve(const unsigned int* input, unsigned int* output, int N) {
    constexpr int kBlockSize = 256;
    constexpr int kBinBits = 8;
    constexpr int kNumBins = 1 << kBinBits;
    int grid_size = (N + kBlockSize - 1) / kBlockSize;

    unsigned int* d_block_histograms;
    cudaMalloc(&d_block_histograms, kNumBins * grid_size * sizeof(unsigned int));

    void* d_scan_storage = nullptr;
    size_t scan_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_scan_storage, scan_storage_bytes, d_block_histograms,
                                  d_block_histograms, kNumBins * grid_size);
    cudaMalloc(&d_scan_storage, scan_storage_bytes);

    unsigned int* d_buffer;
    cudaMalloc(&d_buffer, N * sizeof(unsigned int));
    cudaMemcpy(output, input, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    unsigned int* d_input = output;
    unsigned int* d_output = d_buffer;
    for (unsigned int shift_bits = 0; shift_bits < 32; shift_bits += kBinBits) {
        histogram_kernel<kNumBins>
            <<<grid_size, kBlockSize>>>(d_input, d_block_histograms, N, shift_bits);

        cub::DeviceScan::ExclusiveSum(d_scan_storage, scan_storage_bytes, d_block_histograms,
                                      d_block_histograms, kNumBins * grid_size);

        scatter_kernel<kBlockSize, kNumBins>
            <<<grid_size, kBlockSize>>>(d_input, d_block_histograms, d_output, N, shift_bits);

        std::swap(d_input, d_output);
    }
    if (d_input != output) {
        cudaMemcpy(output, d_input, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    }

    cudaDeviceSynchronize();
    cudaFree(d_buffer);
    cudaFree(d_scan_storage);
    cudaFree(d_block_histograms);
}
