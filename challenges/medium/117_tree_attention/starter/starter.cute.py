import cutlass
import cutlass.cute as cute


# Q, K, V, parents, output are tensors on the GPU
@cute.jit
def solve(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    parents: cute.Tensor,
    output: cute.Tensor,
    T: cute.Int32,
    D: cute.Int32,
):
    pass
