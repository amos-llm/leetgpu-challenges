import torch
import triton
import triton.language as tl


@triton.jit
def jacobi_stencil_kernel(
    input,
    output,
    rows,
    cols,
    stride_ir,
    stride_ic,
    stride_or,
    stride_oc,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_r = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    ptrs_top = input + (offs_r[:, None] - 1) * stride_ir + offs_c[None, :] * stride_ic
    ptrs_bottom = input + (offs_r[:, None] + 1) * stride_ir + offs_c[None, :] * stride_ic
    ptrs_left = input + offs_r[:, None] * stride_ir + (offs_c[None, :] - 1) * stride_ic
    ptrs_right = input + offs_r[:, None] * stride_ir + (offs_c[None, :] + 1) * stride_ic
    mask_edge = (
        (offs_r[:, None] - 1 >= 0)
        & (offs_r[:, None] < rows - 1)
        & (offs_c[None, :] - 1 >= 0)
        & (offs_c[None, :] < cols - 1)
    )
    top = tl.load(ptrs_top, mask=mask_edge)
    bottom = tl.load(ptrs_bottom, mask=mask_edge)
    left = tl.load(ptrs_left, mask=mask_edge)
    right = tl.load(ptrs_right, mask=mask_edge)

    ptrs_center = input + offs_r[:, None] * stride_ir + offs_c[None, :] * stride_ic
    mask_center = (offs_r[:, None] < rows) & (offs_c[None, :] < cols)
    center = tl.load(ptrs_center, mask=mask_center)

    avg = 0.25 * (top + bottom + left + right)
    tile_out = tl.where(mask_edge, avg, center)

    ptrs_out = output + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc
    tl.store(ptrs_out, tile_out, mask=mask_center)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    BLOCK_SIZE_R = 1
    BLOCK_SIZE_C = 1024
    grid = (
        triton.cdiv(rows, BLOCK_SIZE_R),
        triton.cdiv(cols, BLOCK_SIZE_C),
    )
    jacobi_stencil_kernel[grid](
        input,
        output,
        rows,
        cols,
        input.stride(0),
        input.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_R,
        BLOCK_SIZE_C,
    )
