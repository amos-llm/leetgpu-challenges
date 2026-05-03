import torch
import torch.nn.functional as F


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) * torch.rsqrt(var + eps) * gamma + beta


# x, output, weights are tensors on the GPU
def solve(x: torch.Tensor, output: torch.Tensor, weights: torch.Tensor, seq_len: int):
    gamma1 = weights[:768]
    beta1 = weights[768:1536]
    w_qkv = weights[1536:1771008].reshape(768, 2304)
    b_qkv = weights[1771008:1773312]
    w_attn = weights[1773312:2363136].reshape(768, 768)
    b_attn = weights[2363136:2363904]
    gamma2 = weights[2363904:2364672]
    beta2 = weights[2364672:2365440]
    w_fc = weights[2365440:4724736].reshape(768, 3072)
    b_fc = weights[4724736:4727808]
    w_proj = weights[4727808:7087104].reshape(3072, 768)
    b_proj = weights[7087104:]

    resisual = x
    x = layer_norm(x, gamma1, beta1)
    qkv = torch.matmul(x, w_qkv)
    qkv += b_qkv
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.reshape(seq_len, 12, 64).transpose(0, 1)
    k = k.reshape(seq_len, 12, 64).transpose(0, 1)
    v = v.reshape(seq_len, 12, 64).transpose(0, 1)
    s = torch.matmul(q, k.transpose(1, 2)) / 8
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v)
    o = o.transpose(0, 1).reshape(seq_len, 768)
    o = torch.matmul(o, w_attn) + b_attn
    x = o + resisual

    residual = x
    x = layer_norm(x, gamma2, beta2)
    x = torch.matmul(x, w_fc) + b_fc
    x = F.gelu(x, approximate="tanh")
    x = torch.matmul(x, w_proj) + b_proj
    x = x + residual

    output.copy_(x)
