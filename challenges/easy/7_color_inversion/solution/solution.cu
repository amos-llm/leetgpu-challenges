#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int off = threadIdx.x + blockIdx.x * blockDim.x;
    if (off >= width * height) {
        return;
    }
    off *= 4;
    image[off] = 255 - image[off];
    image[off + 1] = 255 - image[off + 1];
    image[off + 2] = 255 - image[off + 2];
}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
