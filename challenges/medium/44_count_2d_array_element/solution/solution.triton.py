import torch
import triton
import triton.language as tl


@triton.jit
def count_kernel(input, output, N, K, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(input + offs, mask=offs < N, other=0.0)
    count = tl.sum(tl.where(x == K, 1, 0))
    tl.atomic_add(output, count, sem="relaxed")


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, M: int, K: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N * M, BLOCK_SIZE),)
    count_kernel[grid](input, output, N * M, K, BLOCK_SIZE)
