import jax
import jax.numpy as jnp


# Q, K_new, V_new, K_cache, V_cache are tensors on GPU
@jax.jit
def solve(
    Q: jax.Array,
    K_new: jax.Array,
    V_new: jax.Array,
    K_cache: jax.Array,
    V_cache: jax.Array,
    seq_len: int,
    H: int,
    D: int,
) -> jax.Array:
    # return output tensor directly
    pass
