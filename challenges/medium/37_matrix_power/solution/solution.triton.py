import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m_tiles = tl.cdiv(M, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = tl.swizzle2d(
        pid // num_n_tiles, pid % num_n_tiles, num_m_tiles, num_n_tiles, GROUP_SIZE
    )

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, K, BLOCK_SIZE_K):
        offs_k = i + tl.arange(0, BLOCK_SIZE_K)
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(
            A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=mask_a, other=0.0
        )
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(
            B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=mask_b, other=0.0
        )
        acc = tl.dot(a, b, acc, allow_tf32=False)

    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc, mask=mask_c)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, P: int):
    if P == 1:
        output.copy_(input)
        return

    input = input.reshape(N, N)
    output = output.reshape(N, N)

    half_power = torch.empty_like(output)
    solve(input, half_power, N, P // 2)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE = 4
    grid = (triton.cdiv(N, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    matmul_kernel[grid](
        half_power,
        half_power,
        output,
        N,
        N,
        N,
        half_power.stride(0),
        half_power.stride(1),
        half_power.stride(0),
        half_power.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE,
    )
    if P % 2 == 1:
        square = torch.empty_like(output)
        square.copy_(output)
        matmul_kernel[grid](
            square,
            input,
            output,
            N,
            N,
            N,
            square.stride(0),
            square.stride(1),
            input.stride(0),
            input.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE,
        )
