import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    input,
    weight,
    bias,
    output,
    C,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_SIZE)
    x = tl.load(input + pid * C + offs_c, mask=offs_c < C, other=0.0)
    w = tl.load(weight + offs_c, mask=offs_c < C, other=0.0)
    b = tl.load(bias + offs_c, mask=offs_c < C, other=0.0)
    sum = tl.sum(x)
    square_sum = tl.sum(x * x)
    mean = sum / C
    variance = (square_sum / C) - mean * mean
    inv_std = tl.rsqrt(variance + eps)
    y = w * (x - mean) * inv_std + b
    tl.store(output + pid * C + offs_c, y, mask=offs_c < C)


# input, weight, bias, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    BLOCK_SIZE_C = triton.next_power_of_2(C)
    grid = (N,)
    layer_norm_kernel[grid](
        input,
        weight,
        bias,
        output,
        C,
        eps,
        BLOCK_SIZE_C,
    )
