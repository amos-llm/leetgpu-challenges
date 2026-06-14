import torch
import triton
import triton.language as tl


@triton.jit
def tanh(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def attn_kernel(
    Q,
    K,
    V,
    output,
    N,
    head_dim: tl.constexpr,
    softcap: tl.constexpr,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_om,
    stride_oh,
    stride_od,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_qo = (offs_m[:, None] < N) & (offs_d[None, :] < head_dim)
    q = tl.load(
        Q + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=mask_qo,
        other=0.0,
    )

    scale = tl.rsqrt(tl.cast(head_dim, tl.float32)) / softcap
    row_max = tl.full((BLOCK_SIZE_M,), -1e38, dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)
    for n in range(0, N, BLOCK_SIZE_N):
        offs_n = n + tl.arange(0, BLOCK_SIZE_N)
        mask_kv = (offs_n[:, None] < N) & (offs_d[None, :] < head_dim)
        k = tl.load(
            K + pid_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=mask_kv,
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k), allow_tf32=False)
        s *= scale
        s = tanh(s)
        s *= softcap
        s = tl.where(offs_n[None, :] < N, s, -1e38)

        row_max_new = tl.maximum(row_max, tl.max(s, axis=1))
        alpha = tl.exp(row_max - row_max_new)

        p = tl.exp(s - row_max_new[:, None])
        v = tl.load(
            V + pid_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=mask_kv,
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=False)

        row_max = row_max_new
        row_sum = row_sum * alpha + tl.sum(p, axis=1)

    acc /= row_sum[:, None]
    tl.store(
        output + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc,
        mask=mask_qo,
    )


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    N: int,
    d_model: int,
    h: int,
    softcap: float,
):
    head_dim = d_model // h
    Q = Q.reshape(N, h, head_dim)
    K = K.reshape(N, h, head_dim)
    V = V.reshape(N, h, head_dim)
    output = output.reshape(N, h, head_dim)

    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_D = max(16, triton.next_power_of_2(head_dim))
    grid = (h, triton.cdiv(N, BLOCK_SIZE_M))
    attn_kernel[grid](
        Q,
        K,
        V,
        output,
        N,
        head_dim,
        softcap,
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
