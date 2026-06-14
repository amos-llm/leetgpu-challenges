#include <cuda_runtime.h>

// Q, K_new, V_new, K_cache, V_cache, output are device pointers
extern "C" void solve(const float* Q, const float* K_new, const float* V_new, float* K_cache,
                      float* V_cache, int seq_len, float* output, int H, int D) {}
