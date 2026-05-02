import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def attn_kernel(
    Q,
    K,
    V,
    output,
    seq_len,
    d_model: tl.constexpr,
    gamma: tl.constexpr,
    stride_qm,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_vn,
    stride_vd,
    stride_om,
    stride_od,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_qo = (offs_m[:, None] < seq_len) & (offs_d[None, :] < d_model)
    q = tl.load(
        Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=mask_qo,
        other=0.0,
    )

    scale_d = tl.rsqrt(tl.cast(d_model, tl.float32))
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)
    for n in range(0, seq_len, BLOCK_SIZE_N):
        offs_n = n + tl.arange(0, BLOCK_SIZE_N)
        mask_kv = (offs_n[:, None] < seq_len) & (offs_d[None, :] < d_model)
        k = tl.load(
            K + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=mask_kv,
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k), allow_tf32=False)
        s *= scale_d

        decay_mask = libdevice.pow(gamma, offs_m[:, None] - offs_n[None, :])
        s *= decay_mask

        causal_mask = offs_m[:, None] >= offs_n[None, :]
        s = tl.where(causal_mask, s, 0.0)

        v = tl.load(
            V + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=mask_kv,
            other=0.0,
        )
        acc += tl.dot(s, v, allow_tf32=False)

    tl.store(
        output + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc,
        mask=mask_qo,
    )


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    seq_len: int,
    d_model: int,
    gamma: float,
):
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_D = max(16, triton.next_power_of_2(d_model))
    grid = (triton.cdiv(seq_len, BLOCK_SIZE_M),)
    attn_kernel[grid](
        Q,
        K,
        V,
        output,
        seq_len,
        d_model,
        gamma,
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
