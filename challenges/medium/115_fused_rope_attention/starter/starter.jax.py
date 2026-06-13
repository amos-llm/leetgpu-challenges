import jax
import jax.numpy as jnp


# Q, K, V, cos, sin are tensors on GPU
@jax.jit
def solve(
    Q: jax.Array,
    K: jax.Array,
    V: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    M: int,
    D: int,
) -> jax.Array:
    # return output tensor directly
    pass
