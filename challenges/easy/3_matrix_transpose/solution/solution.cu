#include <cuda_runtime.h>

template <int kTileSize>
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float s_tile[kTileSize][kTileSize + 1];

    int y_in = blockIdx.y * kTileSize + threadIdx.y;
    int x_in = blockIdx.x * kTileSize + threadIdx.x;
    if (y_in < rows && x_in < cols) {
        s_tile[threadIdx.y][threadIdx.x] = input[y_in * cols + x_in];
    }
    __syncthreads();

    int y_out = blockIdx.x * kTileSize + threadIdx.y;
    int x_out = blockIdx.y * kTileSize + threadIdx.x;
    if (y_out < cols && x_out < rows) {
        output[y_out * rows + x_out] = s_tile[threadIdx.x][threadIdx.y];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    constexpr int kTileSize = 16;
    dim3 block_size(kTileSize, kTileSize);
    dim3 grid_size((cols + kTileSize - 1) / kTileSize, (rows + kTileSize - 1) / kTileSize);
    matrix_transpose_kernel<kTileSize><<<grid_size, block_size>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
