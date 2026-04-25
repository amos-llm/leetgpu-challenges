#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) {
    int off = threadIdx.x + blockIdx.x * blockDim.x;
    if (off >= width * height) {
        return;
    }
    float r = input[off * 3];
    float g = input[off * 3 + 1];
    float b = input[off * 3 + 2];
    float gray = 0.299 * r + 0.587 * g + 0.114 * b;
    output[off] = gray;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}
