import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    Q,
    K,
    V,
    output,
    stride_qm,
    stride_qh,
    stride_qn,
    stride_km,
    stride_kh,
    stride_kn,
    stride_vm,
    stride_vh,
    stride_vn,
    stride_om,
    stride_oh,
    stride_on,
    N,
    d_head: int,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    head = tl.program_id(1)

    offs_m = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_D)

    ptrs_q = Q + offs_m[:, None] * stride_qm + head * stride_qh + offs_d[None, :] * stride_qn
    mask_qo = (offs_m[:, None] < N) & (offs_d[None, :] < d_head)
    q = tl.load(ptrs_q, mask=mask_qo, other=0.0)

    row_max = tl.full((BLOCK_SIZE_M,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)
    scale = 1.0 / tl.sqrt(d_head.to(tl.float32))
    for i in range(0, N, BLOCK_SIZE_N):
        offs_n = i + tl.arange(0, BLOCK_SIZE_N)
        ptrs_k = K + offs_n[:, None] * stride_km + head * stride_kh + offs_d[None, :] * stride_kn
        ptrs_v = V + offs_n[:, None] * stride_vm + head * stride_vh + offs_d[None, :] * stride_vn
        mask_kv = (offs_n[:, None] < N) & (offs_d[None, :] < d_head)
        k = tl.load(ptrs_k, mask=mask_kv, other=0.0)
        v = tl.load(ptrs_v, mask=mask_kv, other=0.0)
        s = tl.dot(q, k.T, allow_tf32=False)
        s *= scale
        s = tl.where(offs_n[None, :] < N, s, float("-inf"))

        row_max_new = tl.maximum(row_max, tl.max(s, axis=1))
        alpha = tl.exp(row_max - row_max_new)

        p = tl.exp(s - row_max_new[:, None])
        o = tl.dot(p, v, allow_tf32=False)

        acc = acc * alpha[:, None] + o
        row_sum = row_sum * alpha + tl.sum(p, axis=1)
        row_max = row_max_new

    acc /= row_sum[:, None]
    ptrs_o = output + offs_m[:, None] * stride_om + head * stride_oh + offs_d[None, :] * stride_on
    tl.store(ptrs_o, acc, mask=mask_qo)


def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    N: int,
    d_model: int,
    h: int,
):
    d_head = d_model // h
    Q = Q.reshape(N, h, d_head)
    K = K.reshape(N, h, d_head)
    V = V.reshape(N, h, d_head)
    output = output.reshape(N, h, d_head)
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_D = max(16, triton.next_power_of_2(d_head))
    grid = (triton.cdiv(N, BLOCK_SIZE_M), h, 1)
    attention_kernel[grid](
        Q,
        K,
        V,
        output,
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
        N,
        d_head,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
