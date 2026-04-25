import torch
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    a,
    b,
    c,
    M,
    N,
    K,
    alpha: float,
    beta: float,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = tl.swizzle2d(pid // num_pid_n, pid % num_pid_n, num_pid_m, num_pid_n, GROUP_SIZE)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    base_offs_k = tl.arange(0, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, K, BLOCK_SIZE_K):
        offs_k = base_offs_k + i
        ptrs_a = a + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        ptrs_b = b + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        block_a = tl.load(ptrs_a, mask=mask_a)
        block_b = tl.load(ptrs_b, mask=mask_b)
        acc = tl.dot(block_a, block_b, acc, allow_tf32=False)

    ptrs_c = c + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    block_c = tl.load(ptrs_c, mask=mask_c)
    block_c = alpha * acc + beta * block_c
    tl.store(ptrs_c, block_c, mask=mask_c)


# a, b, c are tensors on the GPU
def solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
    alpha: float,
    beta: float,
):
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_N = 32
    GROUP_SIZE = 4
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    gemm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        alpha,
        beta,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE,
    )
