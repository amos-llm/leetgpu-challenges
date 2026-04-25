import torch
import triton
import triton.language as tl


@triton.jit
def clip_kernel(input, output, lo, hi, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ptrs_in = input + offs
    mask = offs < N
    block = tl.load(ptrs_in, mask=mask)
    block = tl.where(block < lo, lo, block)
    block = tl.where(block > hi, hi, block)

    ptrs_out = output + offs
    tl.store(ptrs_out, block, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, lo: float, hi: float, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    clip_kernel[grid](input, output, lo, hi, N, BLOCK_SIZE=BLOCK_SIZE)
