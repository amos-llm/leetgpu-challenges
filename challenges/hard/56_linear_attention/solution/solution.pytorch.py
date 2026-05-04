import torch
import torch.nn.functional as F


def phi(x):
    return F.elu(x) + 1


# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):
    p_q = phi(Q)
    p_k = phi(K)
    s = p_q @ (p_k.T @ V)
    z = p_q @ torch.sum(p_k, dim=0)
    output.copy_(s / z[:, None])
