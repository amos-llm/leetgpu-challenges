import torch
import triton
import triton.language as tl


@triton.jit
def prefix_sum_kernel(input, prefix_sum, block_sum, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ptrs_in = input + offs
    mask = offs < N
    x = tl.load(ptrs_in, mask=mask, other=0.0)

    tl.store(prefix_sum + offs, tl.cumsum(x), mask=mask)
    tl.store(block_sum + pid, tl.sum(x))


@triton.jit
def add_offset_kernel(input, offset, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        return

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ptrs_in = input + offs
    mask = offs < N
    x = tl.load(ptrs_in, mask=mask, other=0.0)

    prev_sum = tl.load(offset + pid - 1)
    x += prev_sum
    tl.store(ptrs_in, x, mask=mask)


def prefix_sum(input: torch.Tensor, output: torch.Tensor, N):
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    offset = output.new_empty((num_blocks,))
    grid = (num_blocks,)
    prefix_sum_kernel[grid](input, output, offset, N, BLOCK_SIZE)

    if num_blocks > 1:
        prefix_sum(offset, offset, num_blocks)

    add_offset_kernel[grid](output, offset, N, BLOCK_SIZE)


@triton.jit
def count_kernel(input, output, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(input + offs, mask=offs < N, other=0.0)
    tl.store(output + pid, tl.sum(x > 0))


@triton.jit
def compact_kernel(input, prefix_sum, output, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs_in = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(input + offs_in, mask=offs_in < N, other=0.0)

    if pid > 0:
        start = tl.load(prefix_sum + pid - 1)
    else:
        start = 0
    offs_out = start + tl.cumsum(x > 0) - 1
    tl.store(output + offs_out, x, mask=x > 0)


# A, out are tensors on the GPU
def solve(A: torch.Tensor, N: int, out: torch.Tensor):
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    grid = (num_blocks,)
    offset = out.new_empty((num_blocks,), dtype=torch.int32)
    count_kernel[grid](A, offset, N, BLOCK_SIZE)

    if num_blocks > 1:
        prefix_sum(offset, offset, num_blocks)

    compact_kernel[grid](A, offset, out, N, BLOCK_SIZE)
