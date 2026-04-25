import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ptrs_in = input + offs
    mask = offs < n_elements
    block = tl.load(ptrs_in, mask=mask)
    block = tl.where(block >= 0, block, 0)

    ptrs_out = output + offs
    tl.store(ptrs_out, block, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_kernel[grid](input, output, N, BLOCK_SIZE)
