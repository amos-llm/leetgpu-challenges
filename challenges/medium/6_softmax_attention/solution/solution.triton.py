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
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_D)

    row_max = tl.full((BLOCK_SIZE_M, 1), float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)
    mask_qo = (offs_m[:, None] < M) & (offs_d[None, :] < d)
    q = tl.load(
        Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd, mask=mask_qo, other=0.0
    )
    scale_d = tl.rsqrt(tl.cast(d, tl.float32))
    for i in range(0, N, BLOCK_SIZE_N):
        offs_n = i + tl.arange(0, BLOCK_SIZE_N)
        mask_kv = (offs_n[:, None] < N) & (offs_d[None, :] < d)
        k = tl.load(
            K + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask=mask_kv, other=0.0
        )
        qk = tl.dot(q, tl.trans(k), allow_tf32=False)
        qk *= scale_d
        qk = tl.where(offs_n[None, :] < N, qk, float("-inf"))

        row_max_new = tl.maximum(row_max, tl.max(qk, axis=1, keep_dims=True))
        alpha = tl.exp(row_max - row_max_new)

        p = tl.exp(qk - row_max_new)
        v = tl.load(
            V + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask=mask_kv, other=0.0
        )
        acc = acc * alpha + tl.dot(p, v, allow_tf32=False)

        sum_exp = sum_exp * alpha + tl.sum(p, axis=1, keep_dims=True)
        row_max = row_max_new

    acc /= sum_exp
    tl.store(output + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od, acc, mask=mask_qo)


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    N: int,
    d: int,
):
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_D = max(16, triton.next_power_of_2(d))
    grid = (triton.cdiv(M, BLOCK_SIZE_M),)
    attn_kernel[grid](
        Q,
        K,
        V,
        output,
        M,
        N,
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
