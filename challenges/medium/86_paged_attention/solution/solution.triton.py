import torch
import triton
import triton.language as tl


@triton.jit
def paged_attn_kernel(
    Q,
    K_cache,
    V_cache,
    block_table,
    context_lens,
    output,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_blocks_per_seq,
    stride_qb,
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
    stride_ob,
    stride_oh,
    stride_od,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offs_d = tl.arange(0, head_dim)
    q = tl.load(Q + batch_idx * stride_qb + head_idx * stride_qh + offs_d * stride_qd)
    ctx_len = tl.load(context_lens + batch_idx)

    row_max = -1e38
    row_sum = 0.0
    acc = tl.zeros((head_dim,), dtype=tl.float32)
    sm_scale = tl.rsqrt(tl.cast(head_dim, tl.float32))
    for logical_block_idx in range(tl.cdiv(ctx_len, block_size)):
        offs_t = tl.arange(0, block_size)
        physical_block_idx = tl.load(
            block_table + batch_idx * max_blocks_per_seq + logical_block_idx
        )
        k_block = tl.load(
            K_cache
            + physical_block_idx * stride_kn
            + offs_t[:, None] * stride_kt
            + head_idx * stride_kh
            + offs_d[None, :] * stride_kd,
        )
        s = tl.sum(q[None, :] * k_block, axis=1)
        s *= sm_scale
        s = tl.where(logical_block_idx * block_size + offs_t < ctx_len, s, -1e38)

        row_max_new = tl.maximum(row_max, tl.max(s))
        alpha = tl.exp(row_max - row_max_new)

        p = tl.exp(s - row_max_new)
        v = tl.load(
            V_cache
            + physical_block_idx * stride_vn
            + offs_t[:, None] * stride_vt
            + head_idx * stride_vh
            + offs_d[None, :] * stride_vd,
        )
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)

        row_sum = row_sum * alpha + tl.sum(p)
        row_max = row_max_new

    acc /= row_sum
    tl.store(
        output + batch_idx * stride_ob + head_idx * stride_oh + offs_d * stride_od,
        acc,
    )


# Q, K_cache, V_cache, block_table, context_lens, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    max_blocks_per_seq: int,
):
    grid = (batch_size, num_heads)
    paged_attn_kernel[grid](
        Q,
        K_cache,
        V_cache,
        block_table,
        context_lens,
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
    )
