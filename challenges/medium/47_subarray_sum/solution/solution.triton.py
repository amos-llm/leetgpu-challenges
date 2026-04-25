import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(input, output, S, E, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(input + S + offs, mask=S + offs <= E, other=0.0)
    tl.atomic_add(output, tl.sum(x), sem="relaxed")


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, S: int, E: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv((E - S + 1), BLOCK_SIZE),)
    sum_kernel[grid](input, output, S, E, BLOCK_SIZE)
