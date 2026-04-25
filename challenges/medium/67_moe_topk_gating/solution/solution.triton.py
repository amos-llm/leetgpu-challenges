import torch
import triton
import triton.language as tl


@triton.jit
def moe_topk_kernel(
    logits,
    weights,
    indices,
    M,
    E,
    K: tl.constexpr,
    stride_lm,
    stride_le,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_E: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_e = tl.arange(0, BLOCK_SIZE_E)
    mask_in = (offs_m[:, None] < M) & (offs_e[None, :] < E)
    x = tl.load(
        logits + offs_m[:, None] * stride_lm + offs_e[None, :] * stride_le,
        mask=mask_in,
        other=float("-inf"),
    )

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    topk_weights = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_K), float("-inf"), dtype=tl.float32)
    topk_indices = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.int32)
    for k in range(0, K):
        max_val = tl.max(x, axis=1, keep_dims=True)
        max_index = tl.argmax(x, axis=1, keep_dims=True)
        topk_weights = tl.where(offs_k[None, :] == k, max_val, topk_weights)
        topk_indices = tl.where(offs_k[None, :] == k, max_index, topk_indices)
        x = tl.where(offs_e[None, :] == max_index, float("-inf"), x)

    sum_exp = tl.sum(tl.exp(topk_weights), axis=1, keep_dims=True)
    topk_weights = tl.exp(topk_weights) / sum_exp
    mask_out = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(
        weights + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk,
        topk_weights,
        mask=mask_out,
    )
    tl.store(
        indices + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik,
        topk_indices,
        mask=mask_out,
    )


# logits, topk_weights, topk_indices are tensors on the GPU
def solve(
    logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    M: int,
    E: int,
    k: int,
):
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_E = triton.next_power_of_2(E)
    BLOCK_SIZE_K = triton.next_power_of_2(k)
    grid = (triton.cdiv(M, BLOCK_SIZE_M),)
    moe_topk_kernel[grid](
        logits,
        topk_weights,
        topk_indices,
        M,
        E,
        k,
        logits.stride(0),
        logits.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_indices.stride(0),
        topk_indices.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_E,
        BLOCK_SIZE_K,
    )
