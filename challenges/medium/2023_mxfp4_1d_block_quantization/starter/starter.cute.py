import cutlass
import cutlass.cute as cute


# x, y, scales are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    y: cute.Tensor,
    scales: cute.Tensor,
    M: cute.Int32,
    K: cute.Int32,
):
    pass
