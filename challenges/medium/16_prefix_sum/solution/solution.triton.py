import torch
import triton
import triton.language as tl


@triton.jit
def prefix_sum_kernel(input, prefix_sum, block_sum, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ptrs_in = input + offs
    mask = offs < n
    tile_in = tl.load(ptrs_in, mask=mask, other=0.0)

    tl.store(prefix_sum + offs, tl.cumsum(tile_in), mask=mask)
    tl.store(block_sum + pid, tl.sum(tile_in))


@triton.jit
def add_offset_kernel(input, offset, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        return

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ptrs_in = input + offs
    mask = offs < n
    tile_in = tl.load(ptrs_in, mask=mask, other=0.0)

    prev_sum = tl.load(offset + pid - 1)
    tile_in += prev_sum

    tl.store(ptrs_in, tile_in, mask=mask)


# data and output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, n: int):
    BLOCK_SIZE = 2048
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    block_sum = torch.empty((num_blocks,), dtype=torch.float32, device="cuda")
    grid = (num_blocks,)
    prefix_sum_kernel[grid](input, output, block_sum, n, BLOCK_SIZE)

    if num_blocks > 1:
        solve(block_sum, block_sum, num_blocks)

    add_offset_kernel[grid](output, block_sum, n, BLOCK_SIZE)
