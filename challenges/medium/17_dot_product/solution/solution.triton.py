import torch
import triton
import triton.language as tl


@triton.jit
def dot_product_kernel(a, b, result, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ptrs_a = a + offs
    ptrs_b = b + offs
    mask = offs < n
    c = tl.load(ptrs_a, mask=mask, other=0.0) * tl.load(ptrs_b, mask=mask, other=0.0)
    tl.atomic_add(result, tl.sum(c), sem="relaxed")


# a, b, result are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, result: torch.Tensor, n: int):
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    grid = (num_blocks,)
    dot_product_kernel[grid](a, b, result, n, BLOCK_SIZE)
