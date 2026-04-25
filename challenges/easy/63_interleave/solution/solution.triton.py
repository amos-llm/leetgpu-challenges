import torch
import triton
import triton.language as tl


@triton.jit
def interleave_kernel(A, B, output, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ptrs_A = A + offs
    ptrs_B = B + offs
    mask = offs < N
    block_A = tl.load(ptrs_A, mask=mask)
    block_B = tl.load(ptrs_B, mask=mask)

    tl.store(output + offs * 2, block_A, mask=mask)
    tl.store(output + offs * 2 + 1, block_B, mask=mask)


# A, B, output are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 256

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    interleave_kernel[grid](A, B, output, N, BLOCK_SIZE=BLOCK_SIZE)
