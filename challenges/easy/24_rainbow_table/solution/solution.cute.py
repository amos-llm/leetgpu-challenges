import cutlass.cute as cute


def fnv1a_hash_u32_scalar(x: cute.Uint32) -> cute.Uint32:
    FNV_PRIME = 16777619
    OFFSET_BASIS = 2166136261
    hash_val = cute.Uint32(OFFSET_BASIS)
    prime = cute.Uint32(FNV_PRIME)
    mask = cute.Uint32(0xFF)
    for byte_pos in range(4):
        byte = (x >> (byte_pos * 8)) & mask
        hash_val = (hash_val ^ byte) * prime
    return cute.Uint32(hash_val)


@cute.kernel
def fnv1a_hash_kernel(
    input: cute.Tensor,
    output: cute.Tensor,
    N: cute.Int32,
    R: cute.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    idx = bidx * bdim + tidx
    if idx < N:
        hash = input[idx].to(cute.Uint32)
        while R > 0:
            hash = fnv1a_hash_u32_scalar(hash)
            R -= 1
        output[idx] = hash


# input, output are tensors on the GPU
@cute.jit
def solve(
    input: cute.Tensor,
    output: cute.Tensor,
    N: cute.Int32,
    R: cute.Int32,
):
    threads_per_block = 256
    fnv1a_hash_kernel(input, output, N, R).launch(
        grid=[cute.ceil_div(N, threads_per_block), 1, 1],
        block=[threads_per_block, 1, 1],
    )
