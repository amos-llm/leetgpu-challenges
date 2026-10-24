#include <cuda_runtime.h>
#include <stdint.h>

// Q, K, V, parents, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const int32_t* parents,
                      float* output, int T, int D) {}
