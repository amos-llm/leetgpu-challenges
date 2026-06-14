import jax
import jax.numpy as jnp


# x, y, scale_out are tensors on GPU
@jax.jit
def solve(x: jax.Array, y: jax.Array, scale_out: jax.Array, N: int):
    # return output tensor directly
    pass
