#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void gemm_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                            float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += static_cast<float>(A[row * K + k]) * static_cast<float>(B[k * N + col]);
    }
    C[row * N + col] = static_cast<half>(alpha * acc + beta * static_cast<float>(C[row * N + col]));
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) {
    constexpr int kBlockRows = 16;
    constexpr int kBlockCols = 16;
    dim3 block_size(kBlockCols, kBlockRows);
    dim3 grid_size((N + kBlockCols - 1) / kBlockCols, (M + kBlockRows - 1) / kBlockRows);
    gemm_kernel<<<grid_size, block_size>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}
