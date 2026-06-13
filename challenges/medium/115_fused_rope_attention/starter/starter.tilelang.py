import tilelang
from tilelang import language as T


# Q, K, V, cos, sin, output are tensors on the GPU
@tilelang.jit
def solve(Q: T.Tensor, K: T.Tensor, V: T.Tensor, cos: T.Tensor, sin: T.Tensor, output: T.Tensor, M: int, D: int):
    pass
