#include <cstdint>
#include <cuda_runtime.h>

__device__ float gelu(float x) {
    return 0.5 * x * (1 + erf(x * 0.70710678118));
}

__global__ void geglu_kernel_aligned(const float* input, float* output, int half_n) {
    const float4* up4 = reinterpret_cast<const float4*>(input);
    const float4* gate4 = reinterpret_cast<const float4*>(input + half_n);
    float4* output4 = reinterpret_cast<float4*>(output);

    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    float4 up = up4[vec_idx];
    float4 gate = gate4[vec_idx];
    float4 out;
    out.x = up.x * gelu(gate.x);
    out.y = up.y * gelu(gate.y);
    out.z = up.z * gelu(gate.z);
    out.w = up.w * gelu(gate.w);
    output4[vec_idx] = out;
}

__global__ void geglu_kernel_unaligned(const float* input, float* output, int half_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < half_n) {
        float a = input[idx];
        float b = input[idx + half_n];
        output[idx] = a * gelu(b);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int half_n = N / 2;
    int threadsPerBlock = 256;
    bool is_aligned = (reinterpret_cast<uintptr_t>(input) % 16) == 0 &&
                      (reinterpret_cast<uintptr_t>(output) % 16) == 0 && (half_n % 4) == 0;
    if (is_aligned) {
        int elementsPerBlock = threadsPerBlock * 4;
        int blocksPerGrid = (half_n + elementsPerBlock - 1) / elementsPerBlock;
        geglu_kernel_aligned<<<blocksPerGrid, threadsPerBlock>>>(input, output, half_n);
    } else {
        int blocksPerGrid = (half_n + threadsPerBlock - 1) / threadsPerBlock;
        geglu_kernel_unaligned<<<blocksPerGrid, threadsPerBlock>>>(input, output, half_n);
    }
    cudaDeviceSynchronize();
}
