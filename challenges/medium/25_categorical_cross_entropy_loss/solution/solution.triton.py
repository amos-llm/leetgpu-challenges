import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_loss_kernel(
    logits,
    true_labels,
    loss,
    N,
    C,
    stride_ln,
    stride_lc,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    base_offs_c = tl.arange(0, BLOCK_SIZE_C)

    row_max = tl.full((BLOCK_SIZE_N,), float("-inf"), dtype=tl.float32)
    row_sum_exp = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for i in range(0, C, BLOCK_SIZE_C):
        offs_c = base_offs_c + i
        ptrs_logits = logits + offs_n[:, None] * stride_ln + offs_c[None, :] * stride_lc
        mask_logits = (offs_n[:, None] < N) & (offs_c[None, :] < C)
        block_logits = tl.load(ptrs_logits, mask=mask_logits, other=float("-inf"))
        max_logits = tl.max(block_logits, axis=1)
        row_max_new = tl.maximum(max_logits, row_max)
        row_sum_exp = row_sum_exp * tl.exp(row_max - row_max_new) + tl.sum(
            tl.exp(block_logits - row_max_new[:, None]), axis=1
        )
        row_max = row_max_new

    ptrs_true_labels = true_labels + offs_n[:, None]
    block_true_labels = tl.load(ptrs_true_labels, mask=offs_n[:, None] < N, other=0)

    ptrs_true_logits = logits + offs_n[:, None] * stride_ln + block_true_labels * stride_lc
    true_logits = tl.load(ptrs_true_logits, mask=offs_n[:, None] < N, other=float("-inf"))

    block_loss = -(true_logits - row_max[:, None]) + tl.log(row_sum_exp[:, None])
    block_loss = tl.where(offs_n[:, None] < N, block_loss, 0.0)
    block_loss = block_loss.sum() / N
    tl.atomic_add(loss, block_loss)


# logits, true_labels, loss are tensors on the GPU
def solve(
    logits: torch.Tensor,
    true_labels: torch.Tensor,
    loss: torch.Tensor,
    N: int,
    C: int,
):
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_C = triton.next_power_of_2(C)
    grid = (triton.cdiv(N, BLOCK_SIZE_N),)
    cross_entropy_loss_kernel[grid](
        logits,
        true_labels,
        loss,
        N,
        C,
        logits.stride(0),
        logits.stride(1),
        BLOCK_SIZE_N,
        BLOCK_SIZE_C,
    )
