import torch
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(input, kernel, output, input_size, kernel_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs_in = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ptrs_in = input + offs_in
    ptrs_kernel = kernel
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for _ in range(0, kernel_size):
        mask_in = offs_in < input_size
        block_in = tl.load(ptrs_in, mask=mask_in)
        acc += block_in * tl.load(ptrs_kernel)
        ptrs_in += 1
        ptrs_kernel += 1
        offs_in += 1

    offs_out = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ptrs_out = output + offs_out
    mask_out = offs_out < (input_size - kernel_size + 1)
    tl.store(ptrs_out, acc, mask=mask_out)


# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_size: int,
    kernel_size: int,
):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)

    conv1d_kernel[grid](input, kernel, output, input_size, kernel_size, BLOCK_SIZE)
