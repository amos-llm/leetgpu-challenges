import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid(x):
    return 1 / (1 + tl.exp(-x))


@triton.jit
def sigmoid_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ptrs_in = input + offs
    mask = offs < n_elements
    block = tl.load(ptrs_in, mask=mask)
    block = sigmoid(block)

    ptrs_out = output + offs
    tl.store(ptrs_out, block, mask=mask)


# X, Y are tensors on the GPU
def solve(X: torch.Tensor, Y: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    sigmoid_kernel[grid](X, Y, N, BLOCK_SIZE)
