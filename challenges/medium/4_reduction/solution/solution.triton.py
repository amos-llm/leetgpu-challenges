import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(input, output, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    block = tl.load(input + offs, mask=offs < N, other=0.0)
    sum = tl.sum(block)
    tl.atomic_add(output, sum, sem="relaxed")


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    sum_kernel[grid](input, output, N, BLOCK_SIZE)
