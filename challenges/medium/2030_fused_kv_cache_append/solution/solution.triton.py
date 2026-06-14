import torch
import triton
import triton.language as tl


@triton.jit
def attn_kernel(
    Q,
    K_new,
    V_new,
    K_cache,
    V_cache,
    seq_len,
    output,
    D: tl.constexpr,
    stride_qh,
    stride_qd,
    stride_knh,
    stride_knd,
    stride_vnh,
    stride_vnd,
    stride_kcb,
    stride_kch,
    stride_kcd,
    stride_vcb,
    stride_vch,
    stride_vcd,
    stride_oh,
    stride_od,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    head_idx = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_d = offs_d < D

    tl.store(
        K_cache + seq_len * stride_kcb + head_idx * stride_kch + offs_d * stride_kcd,
        tl.load(K_new + head_idx * stride_knh + offs_d * stride_knd, mask=mask_d),
        mask=mask_d,
    )
    tl.store(
        V_cache + seq_len * stride_vcb + head_idx * stride_vch + offs_d * stride_vcd,
        tl.load(V_new + head_idx * stride_vnh + offs_d * stride_vnd, mask=mask_d),
        mask=mask_d,
    )

    row_max = -1e38
    row_sum = 0.0
    acc = tl.zeros((BLOCK_SIZE_D,), tl.float32)
    q = tl.load(Q + head_idx * stride_qh + offs_d * stride_qd, mask=mask_d, other=0.0)
    for n in range(0, seq_len + 1, BLOCK_SIZE_N):
        offs_n = n + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offs_n < (seq_len + 1)
        mask_load = mask_n[:, None] & mask_d[None, :]
        k = tl.load(
            K_cache
            + offs_n[:, None] * stride_kcb
            + head_idx * stride_kch
            + offs_d[None, :] * stride_kcd,
            mask=mask_load,
            other=0.0,
        )
        s = tl.sum(q[None, :] * k, axis=1)
        s *= tl.rsqrt(tl.cast(D, tl.float32))
        s = tl.where(mask_n, s, -1e38)

        row_max_new = tl.maximum(row_max, tl.max(s, axis=0, return_indices=False))
        alpha = tl.exp(row_max - row_max_new)

        p = tl.exp(s - row_max_new)
        v = tl.load(
            V_cache
            + offs_n[:, None] * stride_vcb
            + head_idx * stride_kch
            + offs_d[None, :] * stride_vcd,
            mask=mask_load,
            other=0.0,
        )
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)

        row_sum = row_sum * alpha + tl.sum(p)
        row_max = row_max_new

    acc /= row_sum
    tl.store(output + head_idx * stride_oh + offs_d * stride_od, acc, mask=mask_d)


# Q, K_new, V_new, K_cache, V_cache, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K_new: torch.Tensor,
    V_new: torch.Tensor,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    seq_len: int,
    output: torch.Tensor,
    H: int,
    D: int,
):
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_D = triton.next_power_of_2(D)
    grid = (H,)
    attn_kernel[grid](
        Q,
        K_new,
        V_new,
        K_cache,
        V_cache,
        seq_len,
        output,
        D,
        Q.stride(0),
        Q.stride(1),
        K_new.stride(0),
        K_new.stride(1),
        V_new.stride(0),
        V_new.stride(1),
        K_cache.stride(0),
        K_cache.stride(1),
        K_cache.stride(2),
        V_cache.stride(0),
        V_cache.stride(1),
        V_cache.stride(2),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_N,
        BLOCK_SIZE_D,
    )
