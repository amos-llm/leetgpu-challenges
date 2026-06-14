import torch
import triton
import triton.language as tl


@triton.jit
def grouped_gemm_kernel(
    A,
    B,
    group_offsets,
    C,
    G,
    K,
    N,
    stride_am,
    stride_ak,
    stride_bg,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    last_problem_end = 0
    for g in range(G):
        m_start = tl.load(group_offsets + g)
        m_end = tl.load(group_offsets + g + 1)
        M_g = m_end - m_start
        num_m_tiles = tl.cdiv(M_g, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        A_g = A + m_start * stride_am
        B_g = B + g * stride_bg
        C_g = C + m_start * stride_cm
        while pid < last_problem_end + num_tiles:
            pid_m, pid_n = tl.swizzle2d(
                (pid - last_problem_end) // num_n_tiles,
                (pid - last_problem_end) % num_n_tiles,
                num_m_tiles,
                num_n_tiles,
                GROUP_SIZE,
            )
            offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, K, BLOCK_SIZE_K):
                offs_k = k + tl.arange(0, BLOCK_SIZE_K)
                mask_a = (offs_m[:, None] < M_g) & (offs_k[None, :] < K)
                a = tl.load(
                    A_g + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=mask_a,
                    other=0.0,
                )
                mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
                b = tl.load(
                    B_g + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=mask_b,
                    other=0.0,
                )
                acc = tl.dot(a, b, acc=acc, allow_tf32=False)
            mask_c = (offs_m[:, None] < M_g) & (offs_n[None, :] < N)
            tl.store(
                C_g + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
                acc,
                mask=mask_c,
            )

            pid += NUM_SM

        last_problem_end += num_tiles


# A, B, group_offsets, C are tensors on the GPU
def solve(
    A: torch.Tensor,
    B: torch.Tensor,
    group_offsets: torch.Tensor,
    C: torch.Tensor,
    G: int,
    M_total: int,
    K: int,
    N: int,
):
    NUM_SM = 40
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    GROUP_SIZE = 4
    grouped_gemm_kernel[(NUM_SM,)](
        A,
        B,
        group_offsets,
        C,
        G,
        K,
        N,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        NUM_SM,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE,
    )
