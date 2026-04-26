#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <int BlockRows, int BlockCols, int MaxKernelRows, int MaxKernelCols>
__global__ void conv2d_kernel(const float* input, const float* kernel, float* output,
                              int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    __shared__ float shared_input[BlockRows + MaxKernelRows][BlockCols + MaxKernelCols];
    for (int i = threadIdx.y; i < BlockRows + MaxKernelRows; i += blockDim.y) {
        for (int j = threadIdx.x; j < BlockCols + MaxKernelCols; j += blockDim.x) {
            int row = blockIdx.y * blockDim.y + i;
            int col = blockIdx.x * blockDim.x + j;
            if (row < input_rows && col < input_cols) {
                shared_input[i][j] = input[row * input_cols + col];
            }
        }
    }
    __syncthreads();

    int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    if (global_row >= output_rows || global_col >= output_cols) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < kernel_rows; ++i) {
        for (int j = 0; j < kernel_cols; ++j) {
            sum += shared_input[threadIdx.y + i][threadIdx.x + j] * kernel[i * kernel_cols + j];
        }
    }
    output[global_row * output_cols + global_col] = sum;
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    constexpr int kBlockRows = 16;
    constexpr int kBlockCols = 16;
    constexpr int kMaxKernelRows = 15;
    constexpr int kMaxKernelCols = 15;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    dim3 block_size(kBlockCols, kBlockRows);
    dim3 grid_size((output_cols + kBlockCols - 1) / kBlockCols,
                   (output_rows + kBlockRows - 1) / kBlockRows);
    int shmem_size =
        (kBlockRows + kernel_rows - 1) * (kBlockCols + kernel_cols - 1) * sizeof(float);
    conv2d_kernel<kBlockRows, kBlockCols, kMaxKernelRows, kMaxKernelCols>
        <<<grid_size, block_size>>>(input, kernel, output, input_rows, input_cols, kernel_rows,
                                    kernel_cols);
}
