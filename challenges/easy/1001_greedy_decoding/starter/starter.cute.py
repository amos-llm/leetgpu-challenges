import cutlass
import cutlass.cute as cute


# logits, tokens are tensors on the GPU
@cute.jit
def solve(logits: cute.Tensor, tokens: cute.Tensor, batch_size: cute.Int32, vocab_size: cute.Int32):
    pass
