import torch
import triton
import triton.language as tl


@triton.jit
def attn_kernel(
    Q,
    K,
    V,
    output,
    M,
    N,
    d: tl.constexpr,
    alpha,
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

    for i in range(0, d, BLOCK_SIZE_D):
        offs_d = i + tl.arange(0, BLOCK_SIZE_D)
        mask_o = (offs_m[:, None] < M) & (offs_d[None, :] < d)
        tl.store(
            output + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
            tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32),
            mask=mask_o,
        )

    row_max = tl.full((BLOCK_SIZE_M, 1), -1e38, dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    for i in range(0, N, BLOCK_SIZE_N):
        offs_n = i + tl.arange(0, BLOCK_SIZE_N)
        s = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for j in range(0, d, BLOCK_SIZE_D):
            offs_d = j + tl.arange(0, BLOCK_SIZE_D)
            mask_q = (offs_m[:, None] < M) & (offs_d[None, :] < d)
            q = tl.load(
                Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd, mask_q, other=0.0
            )
            mask_k = (offs_n[:, None] < N) & (offs_d[None, :] < d)
            k = tl.load(
                K + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=mask_k,
                other=0.0,
            )
            s = tl.dot(q, tl.trans(k), s, allow_tf32=False)

        s *= tl.rsqrt(tl.cast(d, tl.float32))
        s += alpha * (offs_m[:, None] - offs_n[None, :])
        s = tl.where(offs_n[None, :] < N, s, -1e38)

        row_max_new = tl.maximum(row_max, tl.max(s, axis=1, keep_dims=True))
        scale = tl.exp(row_max - row_max_new)

        p = tl.exp(s - row_max_new)
        for j in range(0, d, BLOCK_SIZE_D):
            offs_d = j + tl.arange(0, BLOCK_SIZE_D)
            mask_v = (offs_n[:, None] < N) & (offs_d[None, :] < d)
            v = tl.load(
                V + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=mask_v,
                other=0.0,
            )
            mask_o = (offs_m[:, None] < M) & (offs_d[None, :] < d)
            o = tl.load(
                output + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                mask=mask_o,
                other=0.0,
            )
            o = o * scale + tl.dot(p, v, allow_tf32=False)
            tl.store(
                output + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od, o, mask=mask_o
            )

        row_sum = row_sum * scale + tl.sum(p, axis=1, keep_dims=True)
        row_max = row_max_new

    for i in range(0, d, BLOCK_SIZE_D):
        offs_d = i + tl.arange(0, BLOCK_SIZE_D)
        mask_o = (offs_m[:, None] < M) & (offs_d[None, :] < d)
        o = tl.load(
            output + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
            mask=mask_o,
            other=0.0,
        )
        o /= row_sum
        tl.store(output + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od, o, mask=mask_o)


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    N: int,
    d: int,
    alpha: float,
):
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_D = 64
    grid = (triton.cdiv(M, BLOCK_SIZE_M),)
    attn_kernel[grid](
        Q,
        K,
        V,
        output,
        M,
        N,
        d,
        alpha,
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
