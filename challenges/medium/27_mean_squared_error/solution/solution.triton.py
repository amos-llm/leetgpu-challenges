import torch
import triton
import triton.language as tl


@triton.jit
def mse_kernel(
    predictions,
    targets,
    mse,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    block_p = tl.load(predictions + offs, mask=mask, other=0.0)
    block_t = tl.load(targets + offs, mask=mask, other=0.0)
    error = block_p - block_t
    sum_square = (error * error).sum()
    tl.atomic_add(mse, sum_square / N, sem="relaxed")


# predictions, targets, mse are tensors on the GPU
def solve(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mse: torch.Tensor,
    N: int,
):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    mse_kernel[grid](
        predictions,
        targets,
        mse,
        N,
        BLOCK_SIZE,
        num_warps=8,
    )
