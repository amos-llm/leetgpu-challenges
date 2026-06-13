#include <cuda_runtime.h>
#include <stdint.h>

// x, y, scale_out are device pointers
extern "C" void solve(const float* x, uint8_t* y, float* scale_out, int N) {}
