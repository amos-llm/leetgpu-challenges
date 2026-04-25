import torch
import triton
import triton.language as tl


@triton.jit
def attn_kernel(
    Q,
    K,
    V,
    output,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    seq_len,
    head_dim: tl.constexpr,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_oh,
    stride_om,
    stride_od,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    qo_head_id = tl.program_id(0)
    kv_head_id = qo_head_id // (num_q_heads // num_kv_heads)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_D)

    mask_qo = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    q = tl.load(
        Q + qo_head_id * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=mask_qo,
        other=0.0,
    )

    row_max = tl.full((BLOCK_SIZE_M, 1), -1e38, dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)
    for n in range(0, seq_len, BLOCK_SIZE_N):
        offs_n = n + tl.arange(0, BLOCK_SIZE_N)
        mask_kv = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        k = tl.load(
            K + kv_head_id * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=mask_kv,
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k), allow_tf32=False)
        qk *= tl.rsqrt(tl.cast(head_dim, tl.float32))
        qk = tl.where(offs_n[None, :] < seq_len, qk, -1e38)

        row_max_new = tl.maximum(row_max, tl.max(qk, axis=1, keep_dims=True))
        alpha = tl.exp(row_max - row_max_new)

        p = tl.exp(qk - row_max_new)
        v = tl.load(
            V + kv_head_id * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=mask_kv,
            other=0.0,
        )
        acc = acc * alpha + tl.dot(p, v, allow_tf32=False)

        sum_exp = sum_exp * alpha + tl.sum(p, axis=1, keep_dims=True)
        row_max = row_max_new

    acc /= sum_exp
    tl.store(
        output + qo_head_id * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc,
        mask=mask_qo,
    )


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
):
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_D = max(16, triton.next_power_of_2(head_dim))
    grid = (num_q_heads, triton.cdiv(seq_len, BLOCK_SIZE_M))
    attn_kernel[grid](
        Q,
        K,
        V,
        output,
        num_q_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_D,
    )
