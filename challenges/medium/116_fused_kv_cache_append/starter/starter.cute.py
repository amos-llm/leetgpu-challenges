import cutlass
import cutlass.cute as cute


# Q, K_new, V_new, K_cache, V_cache, output are tensors on the GPU
@cute.jit
def solve(
    Q: cute.Tensor,
    K_new: cute.Tensor,
    V_new: cute.Tensor,
    K_cache: cute.Tensor,
    V_cache: cute.Tensor,
    seq_len: cute.Int32,
    output: cute.Tensor,
    H: cute.Int32,
    D: cute.Int32,
):
    pass
