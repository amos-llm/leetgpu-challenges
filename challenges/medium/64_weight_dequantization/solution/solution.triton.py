import torch
import triton
import triton.language as tl


@triton.jit
def dequant_kernel(
    X,
    S,
    Y,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_sm,
    stride_sn,
    stride_ym,
    stride_yn,
    TILE_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * TILE_SIZE + tl.arange(0, TILE_SIZE)
    offs_n = pid_n * TILE_SIZE + tl.arange(0, TILE_SIZE)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(X + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn, mask=mask)
    s = tl.load(S + pid_m * stride_sm + pid_n * stride_sn)
    y = x * s

    tl.store(Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, y, mask=mask)


# X, S, Y are tensors on the GPU
def solve(
    X: torch.Tensor,
    S: torch.Tensor,
    Y: torch.Tensor,
    M: int,
    N: int,
    TILE_SIZE: int,
):
    grid = (triton.cdiv(M, TILE_SIZE), triton.cdiv(N, TILE_SIZE))
    dequant_kernel[grid](
        X,
        S,
        Y,
        M,
        N,
        X.stride(0),
        X.stride(1),
        S.stride(0),
        S.stride(1),
        Y.stride(0),
        Y.stride(1),
        TILE_SIZE,
    )
