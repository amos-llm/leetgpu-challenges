import tilelang
from tilelang import language as T


# Q, K_new, V_new, K_cache, V_cache, output are tensors on the GPU
@tilelang.jit
def solve(Q: T.Tensor, K_new: T.Tensor, V_new: T.Tensor, K_cache: T.Tensor, V_cache: T.Tensor, seq_len: int, output: T.Tensor, H: int, D: int):
    pass
