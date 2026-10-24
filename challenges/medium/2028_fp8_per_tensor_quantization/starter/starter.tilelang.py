import tilelang
from tilelang import language as T


# x, y, scale_out are tensors on the GPU
@tilelang.jit
def solve(x: T.Tensor, y: T.Tensor, scale_out: T.Tensor, N: int):
    pass
