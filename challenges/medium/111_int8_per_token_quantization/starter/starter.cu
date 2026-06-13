#include <cuda_runtime.h>
#include <stdint.h>

// x, y, scales are device pointers
extern "C" void solve(const float* x, int8_t* y, float* scales, int M, int K) {}
