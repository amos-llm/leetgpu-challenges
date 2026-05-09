import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel_stage1(input, partial_maxs, partial_sums, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(input + offs, mask=offs < N, other=float("-inf"))
    partial_max = tl.max(x)
    partial_sum = tl.sum(tl.exp(x - partial_max))
    tl.store(partial_maxs + pid, partial_max)
    tl.store(partial_sums + pid, partial_sum)


@triton.jit
def softmax_kernel_stage2(input, partial_maxs, partial_sums, output, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    partial_offs = tl.arange(0, BLOCK_SIZE)
    block_max = tl.load(
        partial_maxs + partial_offs, mask=partial_offs < num_programs, other=float("-inf")
    )
    block_sum = tl.load(partial_sums + partial_offs, mask=partial_offs < num_programs, other=0.0)
    row_max = tl.max(block_max)
    row_sum = tl.sum(block_sum * tl.exp(block_max - row_max))
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(input + offs, mask=mask, other=0.0)
    y = tl.exp(x - row_max) / row_sum
    tl.store(output + offs, y, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    partial_maxs = input.new_empty(grid)
    partial_sums = input.new_empty(grid)
    softmax_kernel_stage1[grid](input, partial_maxs, partial_sums, N, BLOCK_SIZE)
    softmax_kernel_stage2[grid](input, partial_maxs, partial_sums, output, N, BLOCK_SIZE)
