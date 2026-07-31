#include <cuda_runtime.h>

__global__ void matrix_add_vectorized(const float* A, const float* B, float* C, int N) {
    const float4* a4 = reinterpret_cast<const float4*>(A);
    const float4* b4 = reinterpret_cast<const float4*>(B);
    float4* c4 = reinterpret_cast<float4*>(C);

    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * N;
    if (vec_idx * 4 + 3 < total_elements) {
        float4 a = a4[vec_idx];
        float4 b = b4[vec_idx];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        c4[vec_idx] = c;
    } else {
        int tail_idx = vec_idx * 4;
        for (int i = 0; tail_idx + i < total_elements; ++i) {
            C[tail_idx + i] = A[tail_idx + i] + B[tail_idx + i];
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
