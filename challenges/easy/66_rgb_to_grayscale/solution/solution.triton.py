import torch
import triton
import triton.language as tl


@triton.jit
def rgb_to_grayscale_kernel(input, output, width, height, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ptrs_in = input + offs * 3
    mask = offs < width * height
    r = tl.load(ptrs_in, mask=mask)
    g = tl.load(ptrs_in + 1, mask=mask)
    b = tl.load(ptrs_in + 2, mask=mask)
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    ptrs_out = output + offs
    tl.store(ptrs_out, gray, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, width: int, height: int):
    total_pixels = width * height
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_pixels, BLOCK_SIZE),)
    rgb_to_grayscale_kernel[grid](input, output, width, height, BLOCK_SIZE)
