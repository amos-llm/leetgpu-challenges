#include <cuda_runtime.h>

__global__ void greedy_decoding_kernel(const float* logits, int* tokens, int batch_size,
                                       int vocab_size) {}

// logits, tokens are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* logits, int* tokens, int batch_size, int vocab_size) {
    // TODO: Set grid and block dimensions and launch kernel
}
