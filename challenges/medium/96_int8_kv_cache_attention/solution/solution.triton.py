import torch
import triton
import triton.language as tl


@triton.jit
def attn_kernel(
    Q,
    K_int8,
    V_int8,
    k_scale,
    v_scale,
    output,
    seq_len,
    head_dim: tl.constexpr,
    stride_qh,
    stride_qd,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ksh,
    stride_ksn,
    stride_vsh,
    stride_vsn,
    stride_oh,
    stride_od,
    BLOCK_SIZE_N: tl.constexpr,
):
    head_id = tl.program_id(0)
    offs_d = tl.arange(0, head_dim)
    q = tl.load(Q + head_id * stride_qh + offs_d * stride_qd)
    row_max = -1e38
    row_sum = 0.0
    acc = tl.zeros((head_dim,), dtype=tl.float32)
    for n in range(0, seq_len, BLOCK_SIZE_N):
        offs_n = n + tl.arange(0, BLOCK_SIZE_N)
        mask_kv = offs_n[:, None] < seq_len
        mask_kvs = offs_n < seq_len
        k = tl.load(
            K_int8
            + head_id * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd,
            mask=mask_kv,
            other=0.0,
        )
        ks = tl.load(k_scale + head_id * stride_ksh + offs_n * stride_ksn, mask=mask_kvs, other=0.0)
        k = tl.cast(k, tl.float32) * ks[:, None]
        s = tl.sum(q[None, :] * k, axis=1)
        s *= tl.rsqrt(tl.cast(head_dim, tl.float32))
        s = tl.where(offs_n < seq_len, s, -1e38)

        row_max_new = tl.maximum(row_max, tl.max(s))
        alpha = tl.exp(row_max - row_max_new)

        p = tl.exp(s - row_max_new)
        v = tl.load(
            V_int8
            + head_id * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd,
            mask=mask_kv,
            other=0.0,
        )
        vs = tl.load(v_scale + head_id * stride_vsh + offs_n * stride_vsn, mask=mask_kvs, other=0.0)
        v = tl.cast(v, tl.float32) * vs[:, None]
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)

        row_sum = row_sum * alpha + tl.sum(p)
        row_max = row_max_new

    acc /= row_sum
    tl.store(output + head_id * stride_oh + offs_d * stride_od, acc)


# Q, K_int8, V_int8, k_scale, v_scale, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K_int8: torch.Tensor,
    V_int8: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    num_heads: int,
    seq_len: int,
    head_dim: int,
):
    BLOCK_SIZE_N = 256
    grid = (num_heads,)
    attn_kernel[grid](
        Q,
        K_int8,
        V_int8,
        k_scale,
        v_scale,
        output,
        seq_len,
        head_dim,
        Q.stride(0),
        Q.stride(1),
        K_int8.stride(0),
        K_int8.stride(1),
        K_int8.stride(2),
        V_int8.stride(0),
        V_int8.stride(1),
        V_int8.stride(2),
        k_scale.stride(0),
        k_scale.stride(1),
        v_scale.stride(0),
        v_scale.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_N,
    )
