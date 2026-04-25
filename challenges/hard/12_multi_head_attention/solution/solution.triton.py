import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    Q,
    K,
    V,
    output,
    stride_qr,
    stride_qh,
    stride_qc,
    stride_kr,
    stride_kh,
    stride_kc,
    stride_vr,
    stride_vh,
    stride_vc,
    stride_or,
    stride_oh,
    stride_oc,
    N,
    d_head: int,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)

    offs_r = row * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    offs_d = tl.arange(0, BLOCK_SIZE_D)

    ptrs_q = Q + offs_r[:, None] * stride_qr + head * stride_qh + offs_d[None, :] * stride_qc
    mask_qo = (offs_r[:, None] < N) & (offs_d[None, :] < d_head)
    q = tl.load(ptrs_q, mask=mask_qo, other=0.0)

    scale = 1.0 / tl.sqrt(d_head.to(tl.float32))

    m_i = tl.full((BLOCK_SIZE_R,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_R,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE_R, BLOCK_SIZE_D), dtype=tl.float32)

    for i in range(0, N, BLOCK_SIZE_C):
        offs_c = i + tl.arange(0, BLOCK_SIZE_C)

        ptrs_k = K + offs_c[:, None] * stride_kr + head * stride_kh + offs_d[None, :] * stride_kc
        ptrs_v = V + offs_c[:, None] * stride_vr + head * stride_vh + offs_d[None, :] * stride_vc
        mask_kv = (offs_c[:, None] < N) & (offs_d[None, :] < d_head)

        k = tl.load(ptrs_k, mask=mask_kv, other=0.0)
        v = tl.load(ptrs_v, mask=mask_kv, other=0.0)

        qk = tl.dot(q, k.T, allow_tf32=False)
        qk *= scale

        qk = tl.where(offs_c[None, :] < N, qk, float("-inf"))

        m_j = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_j)
        alpha = tl.exp(m_i - m_new)

        p = tl.exp(qk - m_new[:, None])
        o = tl.dot(p, v, allow_tf32=False)

        acc = acc * alpha[:, None] + o
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    acc /= l_i[:, None]

    ptrs_o = output + offs_r[:, None] * stride_or + head * stride_oh + offs_d[None, :] * stride_oc
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

    BLOCK_SIZE_R = 8
    BLOCK_SIZE_C = 32
    BLOCK_SIZE_D = max(16, triton.next_power_of_2(d_head))
    grid = (triton.cdiv(N, BLOCK_SIZE_R), h, 1)

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
        BLOCK_SIZE_R=BLOCK_SIZE_R,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
