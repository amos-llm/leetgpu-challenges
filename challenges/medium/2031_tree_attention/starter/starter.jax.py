import jax
import jax.numpy as jnp


# Q, K, V, parents are tensors on GPU
@jax.jit
def solve(
    Q: jax.Array,
    K: jax.Array,
    V: jax.Array,
    parents: jax.Array,
    T: int,
    D: int,
) -> jax.Array:
    # return output tensor directly
    pass
