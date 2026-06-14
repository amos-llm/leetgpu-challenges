#include <cuda_runtime.h>

__global__ void repetition_penalty_kernel(float* logits,        // [B, V]
                                          const int* input_ids, // [B, T]
                                          float penalty, int B, int V, int T) {
    int batch_idx = blockIdx.x;

    __shared__ int bitmap[4096];
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
        bitmap[i] = 0;
    }
    __syncthreads();

    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        int token_id = input_ids[batch_idx * T + t];
        atomicOr(&bitmap[token_id / 32], 1 << (token_id % 32));
    }
    __syncthreads();

    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        if ((bitmap[v / 32] & (1 << (v % 32))) != 0) {
            float logit = logits[batch_idx * V + v];
            if (logit >= 0) {
                logit /= penalty;
            } else {
                logit *= penalty;
            }
            logits[batch_idx * V + v] = logit;
        }
    }
}

// logits, input_ids are device pointers
extern "C" void solve(float* logits,        // [B, V]
                      const int* input_ids, // [B, T]
                      float penalty, int B, int V, int T) {
    constexpr int kBlockSize = 256;
    repetition_penalty_kernel<<<B, kBlockSize>>>(logits, input_ids, penalty, B, V, T);
    cudaDeviceSynchronize();
}
