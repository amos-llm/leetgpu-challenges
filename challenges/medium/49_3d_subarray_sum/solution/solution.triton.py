import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(
    input,
    output,
    S_DEP,
    S_ROW,
    E_ROW,
    S_COL,
    E_COL,
    stride_id,
    stride_ir,
    stride_ic,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_r = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    ptrs_in = (
        input
        + (S_DEP + pid_d) * stride_id
        + (S_ROW + offs_r[:, None]) * stride_ir
        + (S_COL + offs_c[None, :]) * stride_ic
    )
    mask = (S_ROW + offs_r[:, None] <= E_ROW) & (S_COL + offs_c[None, :] <= E_COL)
    x = tl.load(ptrs_in, mask=mask, other=0.0)
    tl.atomic_add(output, tl.sum(x), sem="relaxed")


# input, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    N: int,
    M: int,
    K: int,
    S_DEP: int,
    E_DEP: int,
    S_ROW: int,
    E_ROW: int,
    S_COL: int,
    E_COL: int,
):
    BLOCK_SIZE_D = 1
    BLOCK_SIZE_R = 1
    BLOCK_SIZE_C = 1024
    grid = (
        triton.cdiv((E_DEP - S_DEP + 1), BLOCK_SIZE_D),
        triton.cdiv((E_ROW - S_ROW + 1), BLOCK_SIZE_R),
        triton.cdiv((E_COL - S_COL + 1), BLOCK_SIZE_C),
    )
    sum_kernel[grid](
        input,
        output,
        S_DEP,
        S_ROW,
        E_ROW,
        S_COL,
        E_COL,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        BLOCK_SIZE_R,
        BLOCK_SIZE_C,
    )
