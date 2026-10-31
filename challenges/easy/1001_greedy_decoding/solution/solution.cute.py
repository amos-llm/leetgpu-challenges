import cutlass
import cutlass.cute as cute

BLOCK_SIZE = 256
NUM_WARPS = BLOCK_SIZE // 32


@cute.kernel
def greedy_decoding_kernel(
    logits: cute.Tensor,
    tokens: cute.Tensor,
    vocab_size: cute.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()

    local_max = -1e30
    local_idx = vocab_size

    for idx in range(tidx, vocab_size, bdim):
        logit = logits[(bidx, idx)]
        if logit > local_max:
            local_max = logit
            local_idx = idx
        elif logit == local_max and idx < local_idx:
            local_idx = idx

    for offset in [16, 8, 4, 2, 1]:
        other_max = cute.arch.shuffle_sync_bfly(local_max, offset)
        other_idx = cute.arch.shuffle_sync_bfly(local_idx, offset)
        if other_max > local_max:
            local_max = other_max
            local_idx = other_idx
        elif other_max == local_max and other_idx < local_idx:
            local_idx = other_idx

    allocator = cutlass.utils.SmemAllocator()
    smem_max = allocator.allocate_tensor(cutlass.Float32, cute.make_layout(NUM_WARPS))
    smem_idx = allocator.allocate_tensor(cutlass.Int32, cute.make_layout(NUM_WARPS))
    if lane_idx == 0:
        smem_max[warp_idx] = local_max
        smem_idx[warp_idx] = local_idx
    cute.arch.sync_threads()

    if tidx == 0:
        global_max = smem_max[0]
        global_idx = smem_idx[0]
        for w in cutlass.range_constexpr(1, NUM_WARPS):
            logit = smem_max[w]
            idx = smem_idx[w]
            if logit > global_max:
                global_max = logit
                global_idx = idx
            elif logit == global_max and idx < global_idx:
                global_idx = idx

        tokens[bidx] = global_idx


# logits, tokens are tensors on the GPU
@cute.jit
def solve(logits: cute.Tensor, tokens: cute.Tensor, batch_size: cute.Int32, vocab_size: cute.Int32):
    greedy_decoding_kernel(logits, tokens, vocab_size).launch(
        grid=[batch_size, 1, 1],
        block=[BLOCK_SIZE, 1, 1],
    )
