import tilelang
from tilelang import language as T


# x, y, scales are tensors on the GPU
@tilelang.jit
def solve(x: T.Tensor, y: T.Tensor, scales: T.Tensor, M: int, K: int):
    pass
