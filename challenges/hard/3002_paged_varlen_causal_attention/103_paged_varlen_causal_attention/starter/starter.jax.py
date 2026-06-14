import jax
import jax.numpy as jnp


# Q, K_cache, V_cache, block_table, cu_seqlens are tensors on GPU
@jax.jit
def solve(
    Q: jax.Array,
    K_cache: jax.Array,
    V_cache: jax.Array,
    block_table: jax.Array,
    cu_seqlens: jax.Array,
    T: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    max_blocks_per_seq: int,
    S: int,
) -> jax.Array:
    # return output tensor directly
    pass
