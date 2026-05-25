import cutlass
import cutlass.cute as cute


# Q, K_cache, V_cache, block_table, cu_seqlens, output are tensors on the GPU
@cute.jit
def solve(
    Q: cute.Tensor,
    K_cache: cute.Tensor,
    V_cache: cute.Tensor,
    block_table: cute.Tensor,
    cu_seqlens: cute.Tensor,
    output: cute.Tensor,
    T: cute.Int32,
    num_heads: cute.Int32,
    head_dim: cute.Int32,
    block_size: cute.Int32,
    max_blocks_per_seq: cute.Int32,
    S: cute.Int32,
):
    pass
