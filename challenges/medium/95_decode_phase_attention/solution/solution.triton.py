import torch
import triton
import triton.language as tl


@triton.jit
def attn_kernel(
    Q,
    K,
    V,
    output,
    cache_len,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_od,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_d = tl.arange(0, head_dim)
    q = tl.load(Q + pid_b * stride_qb + pid_h * stride_qh + offs_d * stride_qd)

    row_max = -1e38
    row_sum = 0.0
    acc = tl.zeros((head_dim,), dtype=tl.float32)
    scale_d = tl.rsqrt(tl.cast(head_dim, tl.float32))
    for n in range(0, cache_len, BLOCK_SIZE):
        offs_n = n + tl.arange(0, BLOCK_SIZE)
        mask_kv = offs_n[:, None] < cache_len
        k = tl.load(
            K
            + pid_b * stride_kb
            + pid_h // group_size * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd,
            mask=mask_kv,
            other=0.0,
        )
        s = tl.sum(q[None, :] * k, axis=1)
        s *= scale_d
        s = tl.where(offs_n < cache_len, s, -1e38)

        row_max_new = tl.maximum(row_max, tl.max(s))
        alpha = tl.exp(row_max - row_max_new)

        p = tl.exp(s - row_max_new)
        v = tl.load(
            V
            + pid_b * stride_vb
            + pid_h // group_size * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd,
            mask=mask_kv,
            other=0.0,
        )
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)

        row_sum = row_sum * alpha + tl.sum(p)
        row_max = row_max_new

    acc /= row_sum
    tl.store(output + pid_b * stride_ob + pid_h * stride_oh + offs_d * stride_od, acc)


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    cache_len: int,
    head_dim: int,
):
    BLOCK_SIZE = 32
    grid = (batch_size, num_q_heads)
    attn_kernel[grid](
        Q,
        K,
        V,
        output,
        cache_len,
        head_dim,
        num_q_heads // num_kv_heads,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        V.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_SIZE,
    )
