import torch
import triton
import triton.language as tl


# x, y, scales are tensors on the GPU
def solve(
    x: torch.Tensor,
    y: torch.Tensor,
    scales: torch.Tensor,
    M: int,
    N: int,
    BLOCK_SIZE: int,
):
    pass
