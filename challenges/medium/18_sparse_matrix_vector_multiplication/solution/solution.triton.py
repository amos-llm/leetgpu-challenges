import torch
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(
    A,
    x,
    y,
    stride_am,
    stride_an,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    base_offs_n = tl.arange(0, BLOCK_SIZE_N)
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for i in range(0, N, BLOCK_SIZE_N):
        offs_n = base_offs_n + i
        ptrs_a = A + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
        mask_a = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        ptrs_x = x + offs_n
        mask_x = offs_n < N
        block_a = tl.load(ptrs_a, mask=mask_a, other=0.0)
        block_x = tl.load(ptrs_x, mask=mask_x, other=0.0)
        acc += tl.sum(block_a * block_x[None, :], axis=1)

    ptrs_y = y + offs_m
    mask_y = offs_m < M
    tl.store(ptrs_y, acc, mask=mask_y)


# A, x, y are tensors on the GPU
def solve(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, M: int, N: int, nnz: int):
    A = A.reshape(M, N)
    BLOCK_SIZE_M = 4
    BLOCK_SIZE_N = 512
    grid = (triton.cdiv(M, BLOCK_SIZE_M),)
    gemv_kernel[grid](
        A,
        x,
        y,
        A.stride(0),
        A.stride(1),
        M,
        N,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        num_warps=8,
    )
