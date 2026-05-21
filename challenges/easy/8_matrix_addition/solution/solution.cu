#include <cuda_runtime.h>

__global__ void matrix_add_vectorized(const float* A, const float* B, float* C, int N) {
    auto* a4 = reinterpret_cast<const float4*>(A);
    auto* b4 = reinterpret_cast<const float4*>(B);
    auto* c4 = reinterpret_cast<float4*>(C);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * N;
    if (idx + 3 < total_elements) {
        float4 a = a4[idx];
        float4 b = b4[idx];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        c4[idx] = c;
    } else {
        int float_idx = idx * 4;
        for (int i = 0; i < 4; ++i) {
            if (float_idx + i < total_elements) {
                C[float_idx + i] = A[float_idx + i] + B[float_idx + i];
            }
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int total_elements = N * N;
    int threadsPerBlock = 256;
    int elementsPerBlock = threadsPerBlock * 4;
    int blocksPerGrid = (total_elements + elementsPerBlock - 1) / elementsPerBlock;
    matrix_add_vectorized<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
