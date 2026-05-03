import torch
import torch.nn.functional as F


def rms_norm(x, weight, eps=1e-5):
    return x * weight * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)


def apply_rope(q, cos, sin):
    q1 = q[..., :32]
    q2 = q[..., 32:]
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)


# x, output, weights, cos, sin are tensors on the GPU
def solve(
    x: torch.Tensor,
    output: torch.Tensor,
    weights: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seq_len: int,
):
    w_1 = weights[:512]
    w_qkv = weights[512:393728].reshape(768, 512)
    w_o = weights[393728:655872].reshape(512, 512)
    w_2 = weights[655872:656384]
    w_gate_up = weights[656384:2098176].reshape(2816, 512)
    w_down = weights[2098176:].reshape(512, 1408)

    residual = x
    x = rms_norm(x, weight=w_1)
    qkv = torch.matmul(x, w_qkv.T)
    q, k, v = qkv.split([512, 128, 128], dim=-1)
    q = q.reshape(seq_len, 8, 64)
    k = k.reshape(seq_len, 2, 64)
    v = v.reshape(seq_len, 2, 64)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    k = k.repeat_interleave(4, dim=1)
    v = v.repeat_interleave(4, dim=1)
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    s = torch.matmul(q, k.transpose(1, 2)) / 8
    causal_mask = (
        torch.arange(0, seq_len, device=x.device)[:, None]
        >= torch.arange(0, seq_len, device=x.device)[None, :]
    )
    s = torch.where(causal_mask, s, -float("inf"))
    p = F.softmax(s, dim=-1)  # (num_heads, seq_len, seq_len)
    o = torch.matmul(p, v)  # (num_heads, seq_len, head_dim)
    o = o.transpose(0, 1).reshape(seq_len, -1)  # (seq_len, hidden_size)
    o = torch.matmul(o, w_o.T)  # (seq_len, hidden_size)
    x = residual + o

    residual = x
    x = rms_norm(x, w_2)
    gate_up = torch.matmul(x, w_gate_up.T)
    gate, up = gate_up.chunk(2, dim=-1)
    i = F.silu(gate) * up
    down = torch.matmul(i, w_down.T)
    x = residual + down

    output.copy_(x)
