import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(input, output, N, BLOCK_SIZE: tl.constexpr):
    m_i = float("-inf")
    l_i = 0.0

    offs = tl.arange(0, BLOCK_SIZE)
    ptrs_in = input + offs
    for _ in range(0, N, BLOCK_SIZE):
        mask = offs < N
        block = tl.load(ptrs_in, mask=mask, other=float("-inf"))
        m_j = tl.max(block, axis=0)
        m_new = tl.maximum(m_i, m_j)
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(block - m_new), axis=0)
        m_i = m_new

        offs += BLOCK_SIZE
        ptrs_in += BLOCK_SIZE

    offs = tl.arange(0, BLOCK_SIZE)
    ptrs_in = input + offs
    ptrs_out = output + offs
    for _ in range(0, N, BLOCK_SIZE):
        mask = offs < N
        block = tl.load(ptrs_in, mask=mask, other=float("-inf"))
        block = tl.exp(block - m_i) / l_i
        tl.store(ptrs_out, block, mask=mask)

        offs += BLOCK_SIZE
        ptrs_in += BLOCK_SIZE
        ptrs_out += BLOCK_SIZE


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (1,)

    softmax_kernel[grid](input, output, N, BLOCK_SIZE)
