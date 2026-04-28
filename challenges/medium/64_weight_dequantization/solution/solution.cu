#include <cuda_runtime.h>

__global__ void weight_dequant_kernel(const float* X, const float* S, float* Y, int M, int N,
                                      int TILE_SIZE) {
    int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_row >= M || global_col >= N) {
        return;
    }

    int s_rows = (M + TILE_SIZE - 1) / TILE_SIZE;
    int s_cols = (N + TILE_SIZE - 1) / TILE_SIZE;
    int tile_row = global_row / TILE_SIZE;
    int tile_col = global_col / TILE_SIZE;
    Y[global_row * N + global_col] =
        X[global_row * N + global_col] * S[tile_row * s_cols + tile_col];
}

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
    constexpr int kBlockRows = 32;
    constexpr int kBlockCols = 32;
    dim3 block_size(kBlockCols, kBlockRows);
    dim3 grid_size((N + kBlockCols - 1) / kBlockCols, (M + kBlockRows - 1) / kBlockRows);
    weight_dequant_kernel<<<grid_size, block_size>>>(X, S, Y, M, N, TILE_SIZE);
    cudaDeviceSynchronize();
}
