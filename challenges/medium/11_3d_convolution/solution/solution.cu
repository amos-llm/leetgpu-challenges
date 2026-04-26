#include <cuda_runtime.h>

__global__ void conv3d_kernel(const float* input, const float* kernel, float* output,
                              int input_depth, int input_rows, int input_cols, int kernel_depth,
                              int kernel_rows, int kernel_cols) {
    extern __shared__ float smem_input[];
    int smem_cols = blockDim.x + kernel_cols - 1;
    int smem_rows = blockDim.y + kernel_rows - 1;
    int smem_depth = blockDim.z + kernel_depth - 1;
    for (int i = threadIdx.z; i < smem_depth; i += blockDim.z) {
        for (int j = threadIdx.y; j < smem_rows; j += blockDim.y) {
            for (int k = threadIdx.x; k < smem_cols; k += blockDim.x) {
                int depth = blockIdx.z * blockDim.z + i;
                int row = blockIdx.y * blockDim.y + j;
                int col = blockIdx.x * blockDim.x + k;
                if (depth < input_depth && row < input_rows && col < input_cols) {
                    smem_input[i * smem_rows * smem_cols + j * smem_cols + k] =
                        input[depth * input_rows * input_cols + row * input_cols + col];
                }
            }
        }
    }
    __syncthreads();

    int global_col = blockIdx.x * blockDim.x + threadIdx.x;
    int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    int global_depth = blockIdx.z * blockDim.z + threadIdx.z;
    int output_cols = input_cols - kernel_cols + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int output_depth = input_depth - kernel_depth + 1;

    if (global_depth >= output_depth || global_row >= output_rows || global_col >= output_cols) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < kernel_depth; ++i) {
        for (int j = 0; j < kernel_rows; ++j) {
            for (int k = 0; k < kernel_cols; ++k) {
                sum += smem_input[(threadIdx.z + i) * smem_rows * smem_cols +
                                  (threadIdx.y + j) * smem_cols + threadIdx.x + k] *
                       kernel[i * kernel_rows * kernel_cols + j * kernel_cols + k];
            }
        }
    }
    output[global_depth * output_rows * output_cols + global_row * output_cols + global_col] = sum;
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols) {
    constexpr int kBlockCols = 8;
    constexpr int kBlockRows = 8;
    constexpr int kBlockDepth = 8;
    int output_cols = input_cols - kernel_cols + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int output_depth = input_depth - kernel_depth + 1;
    dim3 block_size(kBlockCols, kBlockRows, kBlockDepth);
    dim3 grid_size((output_cols + kBlockCols - 1) / kBlockCols,
                   (output_rows + kBlockRows - 1) / kBlockRows,
                   (output_depth + kBlockDepth - 1) / kBlockDepth);
    int smem_cols = kBlockCols + kernel_cols - 1;
    int smem_rows = kBlockRows + kernel_rows - 1;
    int smem_depth = kBlockDepth + kernel_depth - 1;
    int smem_size = smem_depth * smem_rows * smem_cols * sizeof(float);
    conv3d_kernel<<<grid_size, block_size, smem_size>>>(input, kernel, output, input_depth,
                                                        input_rows, input_cols, kernel_depth,
                                                        kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
