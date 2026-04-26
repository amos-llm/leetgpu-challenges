#include <cuda_runtime.h>

__global__ void gaussian_blur_kernel(const float* input, const float* kernel, float* output,
                                     int input_rows, int input_cols, int kernel_rows,
                                     int kernel_cols) {
    extern __shared__ float smem_input[];
    int smem_cols = blockDim.x + kernel_cols - 1;
    int smem_rows = blockDim.y + kernel_rows - 1;
    int smem_size = smem_cols * smem_rows * sizeof(float);
    for (int i = threadIdx.y; i < smem_rows; i += blockDim.y) {
        for (int j = threadIdx.x; j < smem_cols; j += blockDim.x) {
            int row = blockIdx.y * blockDim.y + i - kernel_rows / 2;
            int col = blockIdx.x * blockDim.x + j - kernel_cols / 2;
            if (row >= 0 && col >= 0 && row < input_rows && col < input_cols) {
                smem_input[i * smem_cols + j] = input[row * input_cols + col];
            } else {
                smem_input[i * smem_cols + j] = 0.0f;
            }
        }
    }
    __syncthreads();

    int global_col = blockIdx.x * blockDim.x + threadIdx.x;
    int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_row >= input_rows || global_col >= input_cols) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < kernel_rows; ++i) {
        for (int j = 0; j < kernel_cols; ++j) {
            int row = threadIdx.y + i;
            int col = threadIdx.x + j;
            sum += smem_input[row * smem_cols + col] * kernel[i * kernel_cols + j];
        }
    }
    output[global_row * input_cols + global_col] = sum;
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    constexpr int kBlockCols = 16;
    constexpr int kBlockRows = 16;
    dim3 block_size(kBlockCols, kBlockRows);
    dim3 grid_size((input_cols + kBlockCols - 1) / kBlockCols,
                   (input_rows + kBlockRows - 1) / kBlockRows);
    int smem_cols = kBlockCols + kernel_cols - 1;
    int smem_rows = kBlockRows + kernel_rows - 1;
    int smem_size = smem_cols * smem_rows * sizeof(float);
    gaussian_blur_kernel<<<grid_size, block_size, smem_size>>>(
        input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
