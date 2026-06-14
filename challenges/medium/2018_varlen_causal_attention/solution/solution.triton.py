import torch
import triton
import triton.language as tl


@triton.jit
def attn_kernel(
    Q,
    K,
    V,
    cu_seqlens,
    O,
    d: tl.constexpr,
    stride_qt,
    stride_qd,
    stride_kt,
    stride_kd,
    stride_vt,
    stride_vd,
    stride_ot,
    stride_od,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    seq_len = tl.load(cu_seqlens + seq_idx + 1) - tl.load(cu_seqlens + seq_idx)
    pid_m = tl.program_id(1)
    if pid_m * BLOCK_SIZE_M >= seq_len:
        return

    Q_base = Q + tl.load(cu_seqlens + seq_idx) * stride_qt
    K_base = K + tl.load(cu_seqlens + seq_idx) * stride_kt
    V_base = V + tl.load(cu_seqlens + seq_idx) * stride_vt
    O_base = O + tl.load(cu_seqlens + seq_idx) * stride_ot

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_qo = (offs_m[:, None] < seq_len) & (offs_d[None, :] < d)
    q = tl.load(
        Q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
        mask=mask_qo,
        other=0.0,
    )

    row_max = tl.full((BLOCK_SIZE_M,), -1e38, dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)
    sm_scale = tl.rsqrt(tl.cast(d, tl.float32))
    for n in range(0, seq_len, BLOCK_SIZE_N):
        offs_n = n + tl.arange(0, BLOCK_SIZE_N)
        mask_kv = (offs_n[:, None] < seq_len) & (offs_d[None, :] < d)
        k = tl.load(
            K_base + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd,
            mask=mask_kv,
            other=0.0,
        )
        s = tl.dot(q, k.T, allow_tf32=False)
        s *= sm_scale

        mask_attn = (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < seq_len)
        s = tl.where(mask_attn, s, -1e38)

        row_max_new = tl.maximum(row_max, tl.max(s, axis=1))
        alpha = tl.exp(row_max - row_max_new)

        p = tl.exp(s - row_max_new[:, None])
        v = tl.load(
            V_base + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd,
            mask=mask_kv,
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=False)

        row_max = row_max_new
        row_sum = row_sum * alpha + tl.sum(p, axis=1)

    acc /= row_sum[:, None]
    tl.store(
        O_base + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od,
        acc,
        mask=mask_qo,
    )


# Q, K, V, cu_seqlens, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    cu_seqlens: torch.Tensor,
    output: torch.Tensor,
    T: int,
    d: int,
    S: int,
):
    max_seq_len = int(torch.diff(cu_seqlens).max())
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_D = max(16, d)
    grid = (S, triton.cdiv(max_seq_len, BLOCK_SIZE_M))
    attn_kernel[grid](
        Q,
        K,
        V,
        cu_seqlens,
        output,
        d,
        Q.stride(0),
        Q.stride(1),
        K.stride(0),
        K.stride(1),
        V.stride(0),
        V.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_D,
    )
