#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <int kBM, int kBN, int kBK>
__global__ void gemm_kernel(const __half* A, const __half* B, __half* C, int M, int N, int K,
                            float alpha, float beta) {
    __shared__ __half s_A[kBM][kBK];
    __shared__ __half s_B[kBK][kBN];

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += kBK) {
        for (int j = threadIdx.x; j < kBK; j += blockDim.x) {
            int row = blockIdx.y * kBM + threadIdx.y;
            int col = k0 + j;
            if (row < M && col < K) {
                s_A[threadIdx.y][j] = A[row * K + col];
            } else {
                s_A[threadIdx.y][j] = __float2half(0.0f);
            }
        }
        for (int i = threadIdx.y; i < kBK; i += blockDim.y) {
            int row = k0 + i;
            int col = blockIdx.x * kBN + threadIdx.x;
            if (row < K && col < N) {
                s_B[i][threadIdx.x] = B[row * N + col];
            } else {
                s_B[i][threadIdx.x] = __float2half(0.0f);
            }
        }
        __syncthreads();

        for (int k = 0; k < kBK; ++k) {
            acc += __half2float(s_A[threadIdx.y][k]) * __half2float(s_B[k][threadIdx.x]);
        }
        __syncthreads();
    }

    int global_row = blockIdx.y * kBM + threadIdx.y;
    int global_col = blockIdx.x * kBN + threadIdx.x;
    if (global_row < M && global_col < N) {
        float c = __half2float(C[global_row * N + global_col]);
        C[global_row * N + global_col] = alpha * acc + beta * c;
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) {
    constexpr int kBM = 32;
    constexpr int kBN = 32;
    constexpr int kBK = 64;
    dim3 block_size(kBN, kBM);
    dim3 grid_size((N + kBN - 1) / kBN, (M + kBM - 1) / kBM);
    gemm_kernel<kBM, kBN, kBK><<<grid_size, block_size>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}
