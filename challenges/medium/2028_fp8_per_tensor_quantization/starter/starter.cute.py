import cutlass
import cutlass.cute as cute


# x, y, scale_out are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    y: cute.Tensor,
    scale_out: cute.Tensor,
    N: cute.Int32,
):
    pass
