#include <cuda_runtime.h>

// Q, K_cache, V_cache, block_table, cu_seqlens, output are device pointers
extern "C" void solve(const float* Q, const float* K_cache, const float* V_cache,
                      const int* block_table, const int* cu_seqlens, float* output, int T,
                      int num_heads, int head_dim, int block_size, int max_blocks_per_seq, int S) {}
