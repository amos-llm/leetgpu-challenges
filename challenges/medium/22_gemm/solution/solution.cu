#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <int BM, int BN, int BK>
__global__ void gemm_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                            __half* __restrict__ C, int M, int N, int K, float alpha, float beta) {
    __shared__ __half sA[BM * BK];
    __shared__ __half sB[BK * BN];

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int j = threadIdx.x; j < BK; j += blockDim.x) {
            int row = blockIdx.y * BM + threadIdx.y;
            int col = k0 + j;
            __half val = __float2half(0.0f);
            if (row < M && col < K)
                val = A[row * K + col];
            sA[threadIdx.y * BK + j] = val;
        }

        for (int i = threadIdx.y; i < BK; i += blockDim.y) {
            int row = k0 + i;
            int col = blockIdx.x * BN + threadIdx.x;
            __half val = __float2half(0.0f);
            if (row < K && col < N)
                val = B[row * N + col];
            sB[i * BN + threadIdx.x] = val;
        }
        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            float a_val = __half2float(sA[threadIdx.y * BK + k]);
            float b_val = __half2float(sB[k * BN + threadIdx.x]);
            acc += a_val * b_val;
        }
        __syncthreads();
    }

    int global_row = blockIdx.y * BM + threadIdx.y;
    int global_col = blockIdx.x * BN + threadIdx.x;
    if (global_row < M && global_col < N) {
        float c_old = __half2float(C[global_row * N + global_col]);
        float c_new = alpha * acc + beta * c_old;
        C[global_row * N + global_col] = __float2half(c_new);
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
