import torch
import triton
import triton.language as tl


@triton.jit
def topk_kernel(input, output, N, k: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs_in = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(input + offs_in, mask=offs_in < N, other=float("-inf"))
    if k == 1:
        tl.store(output + pid, tl.max(x))
    else:
        offs_out = pid * k + tl.arange(0, k)
        tl.store(output + offs_out, tl.topk(x, k))


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, k: int):
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    grid = (num_blocks,)
    block_topk = input.new_empty((num_blocks, triton.next_power_of_2(k)))
    topk_kernel[grid](input, block_topk, N, triton.next_power_of_2(k), BLOCK_SIZE)
    if num_blocks > 1:
        solve(block_topk, output, block_topk.numel(), k)
    else:
        output[:k].copy_(block_topk[0, :k])
