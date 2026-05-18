#include <cuda_runtime.h>
#include <float.h>

template <int kBlockSizeK>
__global__ void nearest_neighbor_kernel(const float* points, int* indices, int N) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = points[global_idx * 3];
    float y = points[global_idx * 3 + 1];
    float z = points[global_idx * 3 + 2];
    float min_dist = FLT_MAX;
    int min_idx = -1;

    __shared__ float shared_points[kBlockSizeK][3];
    for (int i = 0; i < (N + kBlockSizeK - 1) / kBlockSizeK; ++i) {
        for (int j = threadIdx.x; j < kBlockSizeK; j += blockDim.x) {
            int idx = i * kBlockSizeK + j;
            if (idx < N) {
                shared_points[j][0] = points[idx * 3];
                shared_points[j][1] = points[idx * 3 + 1];
                shared_points[j][2] = points[idx * 3 + 2];
            } else {
                shared_points[j][0] = FLT_MAX;
                shared_points[j][1] = FLT_MAX;
                shared_points[j][2] = FLT_MAX;
            }
        }
        __syncthreads();

#pragma unroll
        for (int j = 0; j < kBlockSizeK; ++j) {
            int idx = i * kBlockSizeK + j;
            if (idx < N && idx != global_idx) {
                float dx = x - shared_points[j][0];
                float dy = y - shared_points[j][1];
                float dz = z - shared_points[j][2];
                float dist = dx * dx + dy * dy + dz * dz;
                if (dist < min_dist) {
                    min_dist = dist;
                    min_idx = idx;
                }
            }
        }
        __syncthreads();
    }

    if (global_idx < N) {
        indices[global_idx] = min_idx;
    }
}

// points and indices are device pointers
extern "C" void solve(const float* points, int* indices, int N) {
    constexpr int kBlockSizeQ = 256;
    constexpr int kBlockSizeK = 256;
    int grid_size = (N + kBlockSizeQ - 1) / kBlockSizeQ;
    nearest_neighbor_kernel<kBlockSizeK><<<grid_size, kBlockSizeQ>>>(points, indices, N);
    cudaDeviceSynchronize();
}
