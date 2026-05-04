import torch
import triton
import triton.language as tl


@triton.jit
def phi(x):
    return tl.where(x > 0, x + 1, tl.exp(x))


@triton.jit
def compute_global_state_kernel(
    K,
    V,
    S,
    Z,
    M,
    d: tl.constexpr,
    stride_km,
    stride_kd,
    stride_vm,
    stride_vd,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_D)

    mask_kv = (offs_m[:, None] < M) & (offs_d[None, :] < d)
    k = tl.load(
        K + offs_m[:, None] * stride_km + offs_d[None, :] * stride_kd,
        mask=mask_kv,
        other=0.0,
    )
    v = tl.load(
        V + offs_m[:, None] * stride_vm + offs_d[None, :] * stride_vd,
        mask=mask_kv,
        other=0.0,
    )

    p_k = phi(k)
    p_k = tl.where(mask_kv, p_k, 0.0)
    s = tl.dot(p_k.T, v, allow_tf32=False)
    z = tl.sum(p_k, axis=0)

    mask_s = (offs_d[:, None] < d) & (offs_d[None, :] < d)
    tl.atomic_add(S + offs_d[:, None] * d + offs_d[None, :], s, mask=mask_s)
    mask_z = offs_d < d
    tl.atomic_add(Z + offs_d, z, mask=mask_z)


@triton.jit
def compute_output_kernel(
    Q,
    S,
    Z,
    output,
    M,
    d: tl.constexpr,
    stride_qm,
    stride_qd,
    stride_om,
    stride_od,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_qo = (offs_m[:, None] < M) & (offs_d[None, :] < d)
    q = tl.load(
        Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=mask_qo,
        other=0.0,
    )
    p_q = phi(q)
    mask_s = (offs_d[:, None] < d) & (offs_d[None, :] < d)
    s = tl.load(S + offs_d[:, None] * d + offs_d[None, :], mask=mask_s, other=0.0)
    mask_z = offs_d < d
    z = tl.load(Z + offs_d, mask=mask_z, other=0.0)

    o = tl.dot(p_q, s, allow_tf32=False) / tl.dot(p_q, z[:, None], allow_tf32=False)
    tl.store(
        output + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        o,
        mask=mask_qo,
    )


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d: int,
):
    BLOCK_SIZE_D = max(16, triton.next_power_of_2(d))
    BLOCK_SIZE_M = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M),)
    S = Q.new_zeros((d, d))
    Z = Q.new_zeros((d,))
    compute_global_state_kernel[grid](
        K,
        V,
        S,
        Z,
        M,
        d,
        K.stride(0),
        K.stride(1),
        V.stride(0),
        V.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_D,
    )
    BLOCK_SIZE_M = 16
    grid = (triton.cdiv(M, BLOCK_SIZE_M),)
    compute_output_kernel[grid](
        Q,
        S,
        Z,
        output,
        M,
        d,
        Q.stride(0),
        Q.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_D,
    )
