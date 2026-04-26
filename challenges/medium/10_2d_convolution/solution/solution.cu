#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* input, const float* kernel, float* output,
                              int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    extern __shared__ float smem_input[];
    int smem_cols = blockDim.x + kernel_cols - 1;
    int smem_rows = blockDim.y + kernel_rows - 1;
    for (int i = threadIdx.y; i < smem_rows; i += blockDim.y) {
        for (int j = threadIdx.x; j < smem_cols; j += blockDim.x) {
            int row = blockIdx.y * blockDim.y + i;
            int col = blockIdx.x * blockDim.x + j;
            if (row < input_rows && col < input_cols) {
                smem_input[i * smem_cols + j] = input[row * input_cols + col];
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
            sum += smem_input[(threadIdx.y + i) * smem_cols + threadIdx.x + j] *
                   kernel[i * kernel_cols + j];
        }
    }
    output[global_row * output_cols + global_col] = sum;
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    constexpr int kBlockCols = 8;
    constexpr int kBlockRows = 8;
    int output_cols = input_cols - kernel_cols + 1;
    int output_rows = input_rows - kernel_rows + 1;
    dim3 block_size(kBlockCols, kBlockRows);
    dim3 grid_size((output_cols + kBlockCols - 1) / kBlockCols,
                   (output_rows + kBlockRows - 1) / kBlockRows);
    int smem_cols = kBlockCols + kernel_cols - 1;
    int smem_rows = kBlockRows + kernel_rows - 1;
    int smem_size = smem_rows * smem_cols * sizeof(float);
    conv2d_kernel<<<grid_size, block_size, smem_size>>>(input, kernel, output, input_rows,
                                                        input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
