import torch
import triton
import triton.language as tl


@triton.jit
def dot_kernel(A, B, result, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    a = tl.load(A + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)
    c = tl.sum(a * b, dtype=tl.float32)
    tl.atomic_add(result, c, sem="relaxed")


# A, B, result are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, result: torch.Tensor, N: int):
    result_fp32 = result.new_zeros((1,), dtype=torch.float32)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    dot_kernel[grid](A, B, result_fp32, N, BLOCK_SIZE)
    result[0] = result_fp32[0].half()
