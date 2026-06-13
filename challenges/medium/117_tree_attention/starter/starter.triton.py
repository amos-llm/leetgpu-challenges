import torch
import triton
import triton.language as tl


# Q, K, V, parents, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    parents: torch.Tensor,
    output: torch.Tensor,
    T: int,
    D: int,
):
    pass
