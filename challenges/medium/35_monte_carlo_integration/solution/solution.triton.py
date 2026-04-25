import torch
import triton
import triton.language as tl


@triton.jit
def mci_kernel(
    y_samples,
    result,
    a,
    b,
    n_samples,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    y = tl.load(y_samples + offs, mask=offs < n_samples, other=0.0)
    tl.atomic_add(result, (b - a) / n_samples * tl.sum(y), sem="relaxed")


# y_samples, result are tensors on the GPU
def solve(
    y_samples: torch.Tensor,
    result: torch.Tensor,
    a: float,
    b: float,
    n_samples: int,
):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_samples, BLOCK_SIZE),)
    mci_kernel[grid](
        y_samples,
        result,
        a,
        b,
        n_samples,
        BLOCK_SIZE,
    )
