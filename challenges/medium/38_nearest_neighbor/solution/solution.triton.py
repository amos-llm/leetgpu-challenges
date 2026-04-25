import torch
import triton
import triton.language as tl


@triton.jit
def nearest_neighbor_kernel(
    points,
    indices,
    N,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_q = pid * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    ptrs_x1 = points + offs_q * 3
    ptrs_y1 = points + offs_q * 3 + 1
    ptrs_z1 = points + offs_q * 3 + 2
    mask_n = offs_q < N
    x1 = tl.load(ptrs_x1, mask=mask_n, other=0.0)
    y1 = tl.load(ptrs_y1, mask=mask_n, other=0.0)
    z1 = tl.load(ptrs_z1, mask=mask_n, other=0.0)

    min_index = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.int32)
    min_dist = tl.full((BLOCK_SIZE_Q,), float("inf"), dtype=tl.float32)
    for i in range(0, N, BLOCK_SIZE_K):
        offs_k = i + tl.arange(0, BLOCK_SIZE_K)
        ptrs_x2 = points + offs_k * 3
        ptrs_y2 = points + offs_k * 3 + 1
        ptrs_z2 = points + offs_k * 3 + 2
        mask_k = offs_k < N
        x2 = tl.load(ptrs_x2, mask=mask_k, other=float("inf"))
        y2 = tl.load(ptrs_y2, mask=mask_k, other=float("inf"))
        z2 = tl.load(ptrs_z2, mask=mask_k, other=float("inf"))
        dx = x1[:, None] - x2[None, :]
        dy = y1[:, None] - y2[None, :]
        dz = z1[:, None] - z2[None, :]
        dist = dx * dx + dy * dy + dz * dz
        dist = tl.where(offs_q[:, None] == offs_k[None, :], float("inf"), dist)
        min_dist_new = tl.min(dist, axis=1)
        min_index = tl.where(min_dist_new < min_dist, i + tl.argmin(dist, axis=1), min_index)
        min_dist = tl.where(min_dist_new < min_dist, min_dist_new, min_dist)

    tl.store(indices + offs_q, min_index, mask=mask_n)


# points and indices are tensors on the GPU
def solve(points: torch.Tensor, indices: torch.Tensor, N: int):
    BLOCK_SIZE_Q = 16
    BLOCK_SIZE_K = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE_Q),)
    nearest_neighbor_kernel[grid](
        points,
        indices,
        N,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_K,
    )
