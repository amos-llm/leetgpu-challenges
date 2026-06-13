import cutlass
import cutlass.cute as cute


# Q, K, V, cos, sin, output are tensors on the GPU
@cute.jit
def solve(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    output: cute.Tensor,
    M: cute.Int32,
    D: cute.Int32,
):
    pass
