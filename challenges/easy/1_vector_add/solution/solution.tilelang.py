import tilelang
from tilelang import language as T


@tilelang.jit
def add_vector(N):
    @T.prim_func
    def main(
        A: T.Tensor[(N,), "float32"],
        B: T.Tensor[(N,), "float32"],
        C: T.Tensor[(N,), "float32"],
    ):
        BLOCK_SIZE = 256
        with T.Kernel(T.ceildiv(N, BLOCK_SIZE), threads=256) as (pid_n,):
            base_n = pid_n * BLOCK_SIZE
            A_frag = T.alloc_fragment((BLOCK_SIZE,), T.float32)
            B_frag = T.alloc_fragment((BLOCK_SIZE,), T.float32)
            C_frag = T.alloc_fragment((BLOCK_SIZE,), T.float32)
            T.copy(A[base_n], A_frag)
            T.copy(B[base_n], B_frag)
            for i in T.Parallel(BLOCK_SIZE):
                C_frag[i] = A_frag[i] + B_frag[i]
            T.copy(C_frag, C[base_n])

    return main


def solve(A, B, C, N):
    kernel = add_vector(N)
    kernel(A, B, C)
