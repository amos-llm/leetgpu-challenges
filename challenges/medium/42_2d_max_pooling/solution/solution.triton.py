import torch
import triton
import triton.language as tl


@triton.jit
def max_pooling_kernel(
    input,
    output,
    H,
    W,
    kernel_size: tl.constexpr,
    stride,
    padding,
    stride_ib,
    stride_ih,
    stride_iw,
    stride_ob,
    stride_oh,
    stride_ow,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    base_offs_h = offs_h * stride - padding
    base_offs_w = offs_w * stride - padding
    max_vals = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_W), float("-inf"), dtype=tl.float32)
    for i in range(0, kernel_size):
        offs_h = base_offs_h + i
        for j in range(0, kernel_size):
            offs_w = base_offs_w + j
            mask = (
                (offs_h[:, None] >= 0)
                & (offs_h[:, None] < H)
                & (offs_w[None, :] >= 0)
                & (offs_w[None, :] < W)
            )
            tile_in = tl.load(
                input
                + pid_b * stride_ib
                + offs_h[:, None] * stride_ih
                + offs_w[None, :] * stride_iw,
                mask=mask,
                other=float("-inf"),
            )
            max_vals = tl.maximum(max_vals, tile_in)

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    mask = (offs_h[:, None] < H_out) & (offs_w[None, :] < W_out)
    tl.store(
        output + pid_b * stride_ob + offs_h[:, None] * stride_oh + offs_w[None, :] * stride_ow,
        max_vals,
        mask=mask,
    )


# input, output are tensors on the GPU
def solve(input, output, N, C, H, W, kernel_size, stride, padding):
    B = N * C
    BLOCK_SIZE_H = 1
    BLOCK_SIZE_W = 256
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    grid = (
        B,
        triton.cdiv(H_out, BLOCK_SIZE_H),
        triton.cdiv(W_out, BLOCK_SIZE_W),
    )
    input = input.reshape(B, H, W)
    output = output.reshape(B, H_out, W_out)
    max_pooling_kernel[grid](
        input,
        output,
        H,
        W,
        kernel_size,
        stride,
        padding,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
    )
