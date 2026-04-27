#include <cuda_runtime.h>

__global__ void stencil2d_kernel(const float* input, float* output, int rows, int cols) {
    extern __shared__ float smem[];
    for (int i = threadIdx.y; i < blockDim.y + 2; i += blockDim.y) {
        for (int j = threadIdx.x; j < blockDim.x + 2; j += blockDim.x) {
            int global_row = blockIdx.y * blockDim.y + i - 1;
            int global_col = blockIdx.x * blockDim.x + j - 1;
            if (global_row >= 0 && global_row < rows && global_col >= 0 && global_col < cols) {
                smem[i * (blockDim.x + 2) + j] = input[global_row * cols + global_col];
            } else {
                smem[i * (blockDim.x + 2) + j] = 0.0f;
            }
        }
    }
    __syncthreads();

    int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_row >= rows || global_col >= cols) {
        return;
    }
    int smem_row = threadIdx.y + 1;
    int smem_col = threadIdx.x + 1;
    if (global_row == 0 || global_row == rows - 1 || global_col == 0 || global_col == cols - 1) {
        output[global_row * cols + global_col] = smem[smem_row * (blockDim.x + 2) + smem_col];
        return;
    }

    float left = smem[smem_row * (blockDim.x + 2) + smem_col - 1];
    float right = smem[smem_row * (blockDim.x + 2) + smem_col + 1];
    float top = smem[(smem_row - 1) * (blockDim.x + 2) + smem_col];
    float bottom = smem[(smem_row + 1) * (blockDim.x + 2) + smem_col];
    output[global_row * cols + global_col] = (left + right + top + bottom) / 4.0f;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    constexpr int kBlockRows = 16;
    constexpr int kBlockCols = 16;
    dim3 block_size(kBlockCols, kBlockRows);
    dim3 grid_size((cols + kBlockCols - 1) / kBlockCols, (rows + kBlockRows - 1) / kBlockRows);
    int smem_size = (kBlockRows + 2) * (kBlockCols + 2) * sizeof(float);
    stencil2d_kernel<<<grid_size, block_size, smem_size>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
