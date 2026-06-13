import jax
import jax.numpy as jnp


# x, y, scales are tensors on GPU
@jax.jit
def solve(x: jax.Array, y: jax.Array, scales: jax.Array, M: int, K: int):
    # return output tensor directly
    pass
