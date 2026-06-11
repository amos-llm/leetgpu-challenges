import jax
import jax.numpy as jnp


# logits is a tensor on the GPU
@jax.jit
def solve(logits: jax.Array, batch_size: int, vocab_size: int) -> jax.Array:
    # return output tensor directly
    pass
