import torch
import triton
import triton.language as tl


@triton.jit
def invert_kernel(image, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    r = tl.load(image + offs, mask=offs < N)
    r = tl.where(offs % 4 != 3, 255 - r, r)
    tl.store(image + offs, r, mask=offs < N)


# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    N = width * height * 4
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    invert_kernel[grid](image, N, BLOCK_SIZE)
