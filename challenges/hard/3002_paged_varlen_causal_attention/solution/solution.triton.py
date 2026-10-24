import torch
import triton
import triton.language as tl


@triton.jit
def attn_kernel(
    Q,
    K_cache,
    V_cache,
    block_table,
    cu_seqlens,
    output,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_blocks_per_seq,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_len = tl.load(cu_seqlens + seq_idx + 1) - tl.load(cu_seqlens + seq_idx)
    pid_m = tl.program_id(2)
    if pid_m * BLOCK_SIZE_M >= seq_len:
        return

    Q_base = Q + tl.load(cu_seqlens + seq_idx) * stride_qt
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_qo = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    q = tl.load(
        Q_base + offs_m[:, None] * stride_qt + head_idx * stride_qh + offs_d[None, :] * stride_qd,
        mask=mask_qo,
        other=0.0,
    )

    row_max = tl.full((BLOCK_SIZE_M,), -1e38, dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)
    sm_scale = tl.rsqrt(tl.cast(head_dim, tl.float32))
    for logical_block_idx in range(tl.cdiv(seq_len, block_size)):
        offs_t = tl.arange(0, block_size)
        physical_block_idx = tl.load(block_table + seq_idx * max_blocks_per_seq + logical_block_idx)
        k = tl.load(
            K_cache
            + physical_block_idx * stride_kn
            + offs_t[:, None] * stride_kt
            + head_idx * stride_kh
            + offs_d[None, :] * stride_kd,
        )
        s = tl.dot(q, tl.trans(k), allow_tf32=False)
        s *= sm_scale
        offs_n = logical_block_idx * block_size + offs_t
        mask_s = (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < seq_len)
        s = tl.where(mask_s, s, -1e38)

        row_max_new = tl.maximum(row_max, tl.max(s, axis=1))
        alpha = tl.exp(row_max - row_max_new)

        p = tl.exp(s - row_max_new[:, None])
        v = tl.load(
            V_cache
            + physical_block_idx * stride_vn
            + offs_t[:, None] * stride_vt
            + head_idx * stride_vh
            + offs_d[None, :] * stride_vd,
        )
        acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=False)

        row_sum = row_sum * alpha + tl.sum(p, axis=1)
        row_max = row_max_new

    acc /= row_sum[:, None]
    output_base = output + tl.load(cu_seqlens + seq_idx) * stride_vt
    tl.store(
        output_base
        + offs_m[:, None] * stride_ot
        + head_idx * stride_oh
        + offs_d[None, :] * stride_od,
        acc,
        mask=mask_qo,
    )


# Q, K_cache, V_cache, block_table, cu_seqlens, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    output: torch.Tensor,
    T: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    max_blocks_per_seq: int,
    S: int,
):
    max_seq_len = int(torch.diff(cu_seqlens).max())
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_D = max(16, head_dim)
    grid = (S, num_heads, triton.cdiv(max_seq_len, BLOCK_SIZE_M))
    attn_kernel[grid](
        Q,
        K_cache,
        V_cache,
        block_table,
        cu_seqlens,
        output,
        head_dim,
        block_size,
        max_blocks_per_seq,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        K_cache.stride(0),
        K_cache.stride(1),
        K_cache.stride(2),
        K_cache.stride(3),
        V_cache.stride(0),
        V_cache.stride(1),
        V_cache.stride(2),
        V_cache.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_SIZE_M,
        BLOCK_SIZE_D,
    )
