import torch
import triton
import triton.language as tl


@triton.jit
def batch_norm_kernel(
    input,
    gamma,
    beta,
    output,
    N,
    C,
    eps,
    stride_in,
    stride_ic,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    sum = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    sum_square = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    for i in range(0, N, BLOCK_SIZE_N):
        offs_n = i + tl.arange(0, BLOCK_SIZE_N)
        mask = (offs_n[:, None] < N) & (offs_c[None, :] < C)
        tile_in = tl.load(
            input + offs_n[:, None] * stride_in + offs_c[None, :] * stride_ic, mask=mask, other=0.0
        )
        sum += tl.sum(tile_in, axis=0)
        sum_square += tl.sum(tile_in * tile_in, axis=0)
    mean = sum / N
    variance = sum_square / N - mean * mean

    for i in range(0, N, BLOCK_SIZE_N):
        offs_n = i + tl.arange(0, BLOCK_SIZE_N)
        mask = (offs_n[:, None] < N) & (offs_c[None, :] < C)
        tile_in = tl.load(
            input + offs_n[:, None] * stride_in + offs_c[None, :] * stride_ic, mask=mask, other=0.0
        )
        x = (tile_in - mean[None, :]) * tl.rsqrt(variance[None, :] + eps)
        g = tl.load(gamma + offs_c, mask=offs_c < C)
        b = tl.load(beta + offs_c, mask=offs_c < C)
        y = g[None, :] * x + b[None, :]
        tl.store(output + offs_n[:, None] * stride_in + offs_c[None, :] * stride_ic, y, mask=mask)


# input, gamma, beta, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_C = 16
    grid = (triton.cdiv(C, BLOCK_SIZE_C),)
    batch_norm_kernel[grid](
        input,
        gamma,
        beta,
        output,
        N,
        C,
        eps,
        input.stride(0),
        input.stride(1),
        BLOCK_SIZE_N,
        BLOCK_SIZE_C,
    )
