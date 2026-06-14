import torch
import triton
import triton.language as tl


@triton.jit
def greedy_decoding_kernel(logits, tokens, batch_size, vocab_size, BLOCK_SIZE: tl.constexpr):
    pass


# logits, tokens are tensors on the GPU
def solve(logits: torch.Tensor, tokens: torch.Tensor, batch_size: int, vocab_size: int):
    grid = (batch_size,)
    greedy_decoding_kernel[grid](logits, tokens, batch_size, vocab_size, BLOCK_SIZE=1024)
