#include <cuda_runtime.h>
#include <stdint.h>

// x, y, scales, global_scale are device pointers
extern "C" void solve(const float* x, uint8_t* y, uint8_t* scales, float* global_scale, int M,
                      int K, int BLOCK_SIZE) {}
