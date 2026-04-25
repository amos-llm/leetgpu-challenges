import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input,
    kernel,
    output,
    stride_im,
    stride_in,
    stride_km,
    stride_kn,
    stride_om,
    stride_on,
    input_rows,
    input_cols,
    kernel_rows: tl.constexpr,
    kernel_cols: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    base_offs_m = pid_m * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    base_offs_n = pid_n * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    acc = tl.zeros((BLOCK_SIZE_R, BLOCK_SIZE_C), dtype=tl.float32)
    for r in range(0, kernel_rows):
        for c in range(0, kernel_cols):
            k = tl.load(kernel + r * stride_km + c * stride_kn)
            offs_m = base_offs_m + r
            offs_n = base_offs_n + c
            ptrs_in = input + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
            mask_in = (offs_m[:, None] < input_rows) & (offs_n[None, :] < input_cols)
            block_in = tl.load(ptrs_in, mask=mask_in, other=0.0)
            acc += block_in * k

    ptrs_out = output + base_offs_m[:, None] * stride_om + base_offs_n[None, :] * stride_on
    mask_out = (base_offs_m[:, None] < input_rows - kernel_rows + 1) & (
        base_offs_n[None, :] < input_cols - kernel_cols + 1
    )
    tl.store(ptrs_out, acc, mask=mask_out)


# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_rows: int,
    input_cols: int,
    kernel_rows: int,
    kernel_cols: int,
):
    input = input.reshape(input_rows, input_cols)
    kernel = kernel.reshape(kernel_rows, kernel_cols)
    output = output.reshape(input_rows - kernel_rows + 1, input_cols - kernel_cols + 1)

    BLOCK_SIZE_R = 64
    BLOCK_SIZE_C = 32
    grid = (
        triton.cdiv(input_rows - kernel_rows + 1, BLOCK_SIZE_R),
        triton.cdiv(input_cols - kernel_cols + 1, BLOCK_SIZE_C),
    )
    conv2d_kernel[grid](
        input,
        kernel,
        output,
        input.stride(0),
        input.stride(1),
        kernel.stride(0),
        kernel.stride(1),
        output.stride(0),
        output.stride(1),
        input_rows,
        input_cols,
        kernel_rows,
        kernel_cols,
        BLOCK_SIZE_R,
        BLOCK_SIZE_C,
    )
