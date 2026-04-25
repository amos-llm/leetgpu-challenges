import torch
import triton
import triton.language as tl


@triton.jit
def sum_square_kernel(
    input,
    sum_square,
    N,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = offs < N

    x = tl.load(input + offs, mask=mask, other=0.0)
    acc = tl.sum(x * x)

    tl.atomic_add(sum_square, acc, sem="relaxed")


@triton.jit
def norm_kernel(
    input,
    sum_square,
    gamma,
    beta,
    output,
    N,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = offs < N

    x = tl.load(input + offs, mask=mask, other=0.0)
    y = x / tl.sqrt(tl.load(sum_square) / N + eps) * gamma + beta
    tl.store(output + offs, y, mask=mask)


# input, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    gamma: float,
    beta: float,
    output: torch.Tensor,
    N: int,
    eps: float,
):
    BLOCK_SIZE_N = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE_N),)

    sum_square = input.new_zeros((1,))

    sum_square_kernel[grid](input, sum_square, N, BLOCK_SIZE_N)

    norm_kernel[grid](input, sum_square, gamma, beta, output, N, eps, BLOCK_SIZE_N)
