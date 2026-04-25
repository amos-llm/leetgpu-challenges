import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(input, output, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ptrs_in = input + offs
    mask = offs < N * N
    block = tl.load(ptrs_in, mask=mask)

    ptrs_out = output + offs
    tl.store(ptrs_out, block, mask=mask)


# a, b are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N * N, BLOCK_SIZE),)
    copy_kernel[grid](a, b, N, BLOCK_SIZE)
