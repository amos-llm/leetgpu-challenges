import torch
import triton
import triton.language as tl


@triton.jit
def reverse_kernel(input, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ptrs = input + offs
    offs_reversed = N - 1 - offs
    ptrs_reversed = input + offs_reversed
    mask = offs < N // 2

    temp = tl.load(ptrs_reversed, mask=mask)
    tl.store(ptrs_reversed, tl.load(ptrs, mask=mask), mask=mask)
    tl.store(ptrs, temp, mask=mask)


# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(N // 2, BLOCK_SIZE)
    grid = (n_blocks,)

    reverse_kernel[grid](input, N, BLOCK_SIZE)
