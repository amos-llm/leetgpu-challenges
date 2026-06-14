#include <cuda_runtime.h>

// Q, K, V, cos, sin, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* cos,
                      const float* sin, float* output, int M, int D) {}
