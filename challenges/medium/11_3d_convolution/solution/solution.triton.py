import torch
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input,
    kernel,
    output,
    stride_id,
    stride_ir,
    stride_ic,
    stride_kd,
    stride_kr,
    stride_kc,
    stride_od,
    stride_or,
    stride_oc,
    input_depth,
    input_rows,
    input_cols,
    kernel_depth: tl.constexpr,
    kernel_rows: tl.constexpr,
    kernel_cols: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)

    base_offs_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    base_offs_r = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    base_offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    acc = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_R, BLOCK_SIZE_C), dtype=tl.float32)
    for d in range(0, kernel_depth):
        for r in range(0, kernel_rows):
            for c in range(0, kernel_cols):
                k = tl.load(kernel + d * stride_kd + r * stride_kr + c * stride_kc)
                offs_d = base_offs_d + d
                offs_r = base_offs_r + r
                offs_c = base_offs_c + c
                ptrs_in = (
                    input
                    + offs_d[:, None, None] * stride_id
                    + offs_r[None, :, None] * stride_ir
                    + offs_c[None, None, :] * stride_ic
                )
                mask_in = (
                    (offs_d[:, None, None] < input_depth)
                    & (offs_r[None, :, None] < input_rows)
                    & (offs_c[None, None, :] < input_cols)
                )
                block_in = tl.load(ptrs_in, mask=mask_in, other=0.0)
                acc += block_in * k

    ptrs_out = (
        output
        + base_offs_d[:, None, None] * stride_od
        + base_offs_r[None, :, None] * stride_or
        + base_offs_c[None, None, :] * stride_oc
    )
    mask_out = (
        (base_offs_d[:, None, None] < input_depth - kernel_depth + 1)
        & (base_offs_r[None, :, None] < input_rows - kernel_rows + 1)
        & (base_offs_c[None, None, :] < input_cols - kernel_cols + 1)
    )
    tl.store(ptrs_out, acc, mask=mask_out)


# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_depth: int,
    input_rows: int,
    input_cols: int,
    kernel_depth: int,
    kernel_rows: int,
    kernel_cols: int,
):
    input = input.reshape(input_depth, input_rows, input_cols)
    kernel = kernel.reshape(kernel_depth, kernel_rows, kernel_cols)
    output = output.reshape(
        input_depth - kernel_depth + 1, input_rows - kernel_rows + 1, input_cols - kernel_cols + 1
    )
    BLOCK_SIZE_D = 4
    BLOCK_SIZE_R = 4
    BLOCK_SIZE_C = 128
    grid = (
        triton.cdiv(input_depth - kernel_depth + 1, BLOCK_SIZE_D),
        triton.cdiv(input_rows - kernel_rows + 1, BLOCK_SIZE_R),
        triton.cdiv(input_cols - kernel_cols + 1, BLOCK_SIZE_C),
    )
    conv3d_kernel[grid](
        input,
        kernel,
        output,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        kernel.stride(0),
        kernel.stride(1),
        kernel.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        input_depth,
        input_rows,
        input_cols,
        kernel_depth,
        kernel_rows,
        kernel_cols,
        BLOCK_SIZE_D,
        BLOCK_SIZE_R,
        BLOCK_SIZE_C,
    )
