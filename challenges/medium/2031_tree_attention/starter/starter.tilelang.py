import tilelang
from tilelang import language as T


# Q, K, V, parents, output are tensors on the GPU
@tilelang.jit
def solve(
    Q: T.Tensor,
    K: T.Tensor,
    V: T.Tensor,
    parents: T.Tensor,
    output: T.Tensor,
    T: int,
    D: int,
):
    pass
