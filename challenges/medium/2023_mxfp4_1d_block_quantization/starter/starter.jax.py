import jax
import jax.numpy as jnp


# x, y, scales are tensors on GPU
@jax.jit
def solve(X: jax.Array, Y: jax.Array, scales: jax.Array, M: int, K: int) -> jax.Array:
    # return output tensor directly
    pass
