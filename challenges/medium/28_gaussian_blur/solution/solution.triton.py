import torch
import triton
import triton.language as tl


@triton.jit
def gaussian_blur_kernel(
    input,
    kernel,
    output,
    input_rows,
    input_cols,
    kernel_rows: tl.constexpr,
    kernel_cols: tl.constexpr,
    stride_ir,
    stride_ic,
    stride_kr,
    stride_kc,
    stride_or,
    stride_oc,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_r = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    acc = tl.zeros((BLOCK_SIZE_R, BLOCK_SIZE_C), dtype=tl.float32)
    for i in range(0, kernel_rows):
        for j in range(0, kernel_cols):
            ptrs_in = (
                input
                + (offs_r[:, None] - kernel_rows // 2 + i) * stride_ir
                + (offs_c[None, :] - kernel_cols // 2 + j) * stride_ic
            )
            mask_in = (
                ((offs_r[:, None] - kernel_rows // 2 + i) >= 0)
                & ((offs_r[:, None] - kernel_rows // 2 + i) < input_rows)
                & ((offs_c[None, :] - kernel_cols // 2 + j) >= 0)
                & ((offs_c[None, :] - kernel_cols // 2 + j) < input_cols)
            )
            block_in = tl.load(ptrs_in, mask=mask_in, other=0.0)
            k = tl.load(kernel + i * stride_kr + j * stride_kc)
            acc += block_in * k

    ptrs_out = output + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc
    mask_out = (offs_r[:, None] < input_rows) & (offs_c[None, :] < input_cols)
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
    output = output.reshape(input_rows, input_cols)
    BLOCK_SIZE_R = 32
    BLOCK_SIZE_C = 32
    grid = (
        triton.cdiv(input_rows, BLOCK_SIZE_R),
        triton.cdiv(input_cols, BLOCK_SIZE_C),
    )
    gaussian_blur_kernel[grid](
        input,
        kernel,
        output,
        input_rows,
        input_cols,
        kernel_rows,
        kernel_cols,
        input.stride(0),
        input.stride(1),
        kernel.stride(0),
        kernel.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_R,
        BLOCK_SIZE_C,
    )
