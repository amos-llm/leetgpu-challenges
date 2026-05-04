import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
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

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=mask_a)
        b = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=mask_b)
        acc = tl.dot(a, b, acc, allow_tf32=False)

    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=mask_c,
    )


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def gated_matmul_kernel(
    A,
    B0,
    B1,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_b0k,
    stride_b0n,
    stride_b1k,
    stride_b1n,
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

    acc0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(
            A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_a,
            other=0.0,
        )
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b0 = tl.load(
            B0 + offs_k[:, None] * stride_b0k + offs_n[None, :] * stride_b0n,
            mask=mask_b,
            other=0.0,
        )
        b1 = tl.load(
            B1 + offs_k[:, None] * stride_b1k + offs_n[None, :] * stride_b1n,
            mask=mask_b,
            other=0.0,
        )
        acc0 = tl.dot(a, b0, acc=acc0, allow_tf32=False)
        acc1 = tl.dot(a, b1, acc=acc1, allow_tf32=False)

    c = silu(acc0) * acc1
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=mask_c,
    )


# x, W_gate, W_up, W_down, output are tensors on the GPU
def solve(
    x: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d_model: int,
    d_ffn: int,
):
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_N = 64
    GROUP_SIZE = 4
    grid_up = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(d_ffn, BLOCK_SIZE_N),)
    intermediate = x.new_empty((M, d_ffn))
    gated_matmul_kernel[grid_up](
        x,
        W_gate,
        W_up,
        intermediate,
        M,
        d_ffn,
        d_model,
        x.stride(0),
        x.stride(1),
        W_gate.stride(0),
        W_gate.stride(1),
        W_up.stride(0),
        W_up.stride(1),
        intermediate.stride(0),
        intermediate.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE,
    )
    grid_down = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(d_model, BLOCK_SIZE_N),)
    matmul_kernel[grid_down](
        intermediate,
        W_down,
        output,
        M,
        d_model,
        d_ffn,
        intermediate.stride(0),
        intermediate.stride(1),
        W_down.stride(0),
        W_down.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE,
    )
