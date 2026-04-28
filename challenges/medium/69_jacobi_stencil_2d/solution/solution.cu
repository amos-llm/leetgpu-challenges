#include <cuda_runtime.h>

__global__ void stencil2d_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) {
        return;
    }
    if (row == 0 || row == rows - 1 || col == 0 || col == cols - 1) {
        output[row * cols + col] = input[row * cols + col];
        return;
    }
    float left = input[row * cols + col - 1];
    float right = input[row * cols + col + 1];
    float top = input[(row - 1) * cols + col];
    float bottom = input[(row + 1) * cols + col];
    output[row * cols + col] = (left + right + top + bottom) / 4.0f;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    constexpr int kBlockRows = 16;
    constexpr int kBlockCols = 16;
    dim3 block_size(kBlockCols, kBlockRows);
    dim3 grid_size((cols + kBlockCols - 1) / kBlockCols, (rows + kBlockRows - 1) / kBlockRows);
    stencil2d_kernel<<<grid_size, block_size>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
