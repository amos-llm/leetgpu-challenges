import torch
import torch.nn.functional as F


# x, W_gate, W_up, W_down, output are tensors on the GPU
def solve(
    x: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d_model: int,
    d_ffn: int,
):
    gate = x @ W_gate
    up = x @ W_up
    x = F.silu(gate) * up
    x = x @ W_down
    output.copy_(x)
