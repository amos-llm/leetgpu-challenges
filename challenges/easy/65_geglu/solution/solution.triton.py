import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return 0.5 * x * (1 + tl.erf(x * 0.70710678118))


@triton.jit
def geglu(input, output, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ptrs_x1 = input + offs
    ptrs_x2 = ptrs_x1 + N // 2
    mask = offs < N // 2
    block_x1 = tl.load(ptrs_x1, mask=mask)
    block_x2 = tl.load(ptrs_x2, mask=mask)
    block_out = block_x1 * gelu(block_x2)

    ptrs_out = output + offs
    tl.store(ptrs_out, block_out, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE),)
    geglu[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)
