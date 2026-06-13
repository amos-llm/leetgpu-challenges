import torch


# Q, K_new, V_new, K_cache, V_cache, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K_new: torch.Tensor,
    V_new: torch.Tensor,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    seq_len: int,
    output: torch.Tensor,
    H: int,
    D: int,
):
    pass
