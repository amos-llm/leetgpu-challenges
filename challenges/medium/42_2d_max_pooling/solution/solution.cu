#include <cuda_runtime.h>

__global__ void max_pooling_kernel(const float* input, float* output, int B, int H, int W,
                                   int kernel_size, int stride, int padding) {
    int output_cols = (W + 2 * padding - kernel_size) / stride + 1;
    int output_rows = (H + 2 * padding - kernel_size) / stride + 1;
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;
    int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_col >= output_cols || global_row >= output_rows) {
        return;
    }

    float max_val = -1e38;
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int col = global_col * stride + j - padding;
            int row = global_row * stride + i - padding;
            if (col >= 0 && col < W && row >= 0 && row < H) {
                max_val = max(max_val, input[blockIdx.z * H * W + row * W + col]);
            }
        }
    }
    output[blockIdx.z * output_rows * output_cols + global_row * output_cols + global_col] =
        max_val;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {
    constexpr int kBlockCols = 16;
    constexpr int kBlockRows = 16;
    int output_cols = (W + 2 * padding - kernel_size) / stride + 1;
    int output_rows = (H + 2 * padding - kernel_size) / stride + 1;
    dim3 block_size(kBlockCols, kBlockRows);
    dim3 grid_size((output_cols + kBlockCols - 1) / kBlockCols,
                   (output_rows + kBlockRows - 1) / kBlockRows, N * C);
    max_pooling_kernel<<<grid_size, block_size>>>(input, output, N * C, H, W, kernel_size, stride,
                                                  padding);
    cudaDeviceSynchronize();
}
