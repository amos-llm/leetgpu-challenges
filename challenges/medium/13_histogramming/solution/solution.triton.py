import torch
import triton
import triton.language as tl


@triton.jit
def hist_kernel(
    input,
    histogram,
    N,
    num_bins,
    BLOCK_SIZE: tl.constexpr,
    NUM_BINS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_in = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ptrs_in = input + offs_in
    mask_in = offs_in < N
    block = tl.load(ptrs_in, mask=mask_in)
    hist = tl.histogram(block, NUM_BINS, mask=mask_in)

    offs_hist = tl.arange(0, NUM_BINS)
    mask_hist = offs_hist < num_bins
    tl.atomic_add(histogram + offs_hist, hist, mask=mask_hist)


# input, histogram are tensors on the GPU
def solve(input: torch.Tensor, histogram: torch.Tensor, N: int, num_bins: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    NUM_BINS = triton.next_power_of_2(num_bins)
    hist_kernel[grid](
        input,
        histogram,
        N,
        num_bins,
        BLOCK_SIZE,
        NUM_BINS,
    )
