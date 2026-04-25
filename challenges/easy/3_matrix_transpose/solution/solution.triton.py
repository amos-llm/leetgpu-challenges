import torch
import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(
    input,
    output,
    rows,
    cols,
    stride_im,
    stride_in,
    stride_on,
    stride_om,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x = tl.load(
        input + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in,
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < cols),
    )
    tl.store(
        output + offs_n[:, None] * stride_on + offs_m[None, :] * stride_om,
        tl.trans(x),
        mask=(offs_n[:, None] < cols) & (offs_m[None, :] < rows),
    )


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    grid = (triton.cdiv(rows, BLOCK_SIZE_M), triton.cdiv(cols, BLOCK_SIZE_N))
    matrix_transpose_kernel[grid](
        input,
        output,
        rows,
        cols,
        input.stride(0),
        input.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
