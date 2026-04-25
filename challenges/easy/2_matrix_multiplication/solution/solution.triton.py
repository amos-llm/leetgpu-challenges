import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a,
    b,
    c,
    M,
    N,
    K,
    stride_am,
    stride_an,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_ck,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    pid_m, pid_k = tl.swizzle2d(pid // num_pid_k, pid % num_pid_k, num_pid_m, num_pid_k, GROUP_SIZE)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    base_offs_n = tl.arange(0, BLOCK_SIZE_N)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for i in range(0, N, BLOCK_SIZE_N):
        offs_n = base_offs_n + i
        ptrs_a = a + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
        ptrs_b = b + offs_k[None, :] * stride_bk + offs_n[:, None] * stride_bn
        mask_a = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        mask_b = (offs_k[None, :] < K) & (offs_n[:, None] < N)
        block_a = tl.load(ptrs_a, mask=mask_a)
        block_b = tl.load(ptrs_b, mask=mask_b)
        acc = tl.dot(block_a, block_b, acc, allow_tf32=False)

    ptrs_c = c + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck
    mask_c = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(ptrs_c, acc, mask=mask_c)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_N = 32
    GROUP_SIZE = 4
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(K, BLOCK_SIZE_K),)
    matrix_multiplication_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        stride_am,
        stride_an,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_ck,
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
        BLOCK_SIZE_N,
        GROUP_SIZE,
    )
