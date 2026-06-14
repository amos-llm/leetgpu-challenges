import torch
import triton
import triton.language as tl


@triton.jit
def rms_kernel(
    input,
    residual,
    weight,
    output,
    C,
    eps,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    mask = offs < C
    x = tl.load(input + pid_n * C + offs, mask=mask, other=0.0)
    r = tl.load(residual + pid_n * C + offs, mask=mask, other=0.0)
    z = x + r
    rrms = tl.rsqrt(tl.sum(z * z, 0) / C + eps)
    w = tl.load(weight + offs, mask=mask, other=0.0)
    y = z * rrms * w
    tl.store(output + pid_n * C + offs, y, mask=mask)


# x, residual, weight, out are tensors on the GPU
def solve(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    BLOCK_SIZE_C = triton.next_power_of_2(C)
    grid = (N, triton.cdiv(C, BLOCK_SIZE_C))
    rms_kernel[grid](x, residual, weight, out, C, eps, BLOCK_SIZE_C)
