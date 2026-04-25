import torch
import triton
import triton.language as tl


@triton.jit
def fnv1a_hash(x):
    FNV_PRIME = 16777619
    OFFSET_BASIS = 2166136261

    hash_val = tl.full(x.shape, OFFSET_BASIS, tl.uint32)

    for byte_pos in range(4):
        byte = (x >> (byte_pos * 8)) & 0xFF
        hash_val = (hash_val ^ byte) * FNV_PRIME

    return hash_val


@triton.jit
def fnv1a_hash_kernel(input, output, n_elements, n_rounds, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ptrs_in = input + offs
    mask = offs < n_elements
    block = tl.load(ptrs_in, mask=mask)
    block = block.to(tl.uint32)
    for _ in range(0, n_rounds):
        block = fnv1a_hash(block)

    ptrs_out = output + offs
    tl.store(ptrs_out, block, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, R: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    fnv1a_hash_kernel[grid](input, output, N, R, BLOCK_SIZE)
