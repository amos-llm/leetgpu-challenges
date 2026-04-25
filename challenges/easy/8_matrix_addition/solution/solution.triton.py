import torch
import triton
import triton.language as tl


@triton.jit
def matrix_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ptrs_a = a + offs
    ptrs_b = b + offs
    mask = offs < n_elements
    block_a = tl.load(ptrs_a, mask=mask)
    block_b = tl.load(ptrs_b, mask=mask)

    ptrs_c = c + offs
    tl.store(ptrs_c, block_a + block_b, mask=mask)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_elements = N * N
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    matrix_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)
