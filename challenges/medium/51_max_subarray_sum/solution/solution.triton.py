import torch
import triton
import triton.language as tl


@triton.jit
def prefix_sum_kernel(input, prefix_sum, block_sum, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tile_in = tl.load(input + offs, mask=offs < N, other=0.0)

    tl.store(prefix_sum + offs, tl.cumsum(tile_in), mask=offs < N)
    tl.store(block_sum + pid, tl.sum(tile_in))


@triton.jit
def add_offset_kernel(prefix_sum, offset, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        return

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tile_in = tl.load(prefix_sum + offs, mask=offs < N, other=0.0)

    tl.store(prefix_sum + offs, tile_in + tl.load(offset + pid - 1), mask=offs < N)


def prefix_sum(input: torch.Tensor, output: torch.Tensor, N):
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    block_sum = input.new_empty((num_blocks,))
    grid = (num_blocks,)
    prefix_sum_kernel[grid](input, output, block_sum, N, BLOCK_SIZE)
    if num_blocks > 1:
        prefix_sum(block_sum, block_sum, num_blocks)
    add_offset_kernel[grid](output, block_sum, N, BLOCK_SIZE)


@triton.jit
def max_sum_kernel(input, output, N, window_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    win_start = tl.load(input + offs, mask=offs < N - window_size + 1, other=500000)
    win_end = tl.load(input + offs + window_size, mask=offs + window_size < N, other=-500000)
    tl.atomic_max(output, tl.max(win_end - win_start), sem="relaxed")


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, window_size: int):
    prefix_sum(input, input, N)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N - window_size + 1, BLOCK_SIZE),)
    output[0] = -500000
    max_sum_kernel[grid](input, output, N, window_size, BLOCK_SIZE)
