import tilelang
import tilelang.language as T
import torch


@tilelang.jit
def matrix_transpose(rows, cols):
    @T.prim_func
    def main(
        input: T.Tensor[(rows, cols), T.float32],
        output: T.Tensor[(cols, rows), T.float32],
    ):
        BLOCK_SIZE = 16
        with T.Kernel(
            T.ceildiv(rows, BLOCK_SIZE),
            T.ceildiv(cols, BLOCK_SIZE),
            threads=256,
        ) as (
            pid_m,
            pid_n,
        ):
            base_m = pid_m * BLOCK_SIZE
            base_n = pid_n * BLOCK_SIZE
            shared_in = T.alloc_shared((BLOCK_SIZE, BLOCK_SIZE), T.float32)
            shared_out = T.alloc_shared((BLOCK_SIZE, BLOCK_SIZE), T.float32)
            T.copy(input[base_m, base_n], shared_in)
            for i, j in T.Parallel(BLOCK_SIZE, BLOCK_SIZE):
                shared_out[i, j] = shared_in[j, i]
            T.copy(shared_out, output[base_n, base_m])

    return main


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    kernel = matrix_transpose(rows, cols)
    kernel(input, output)
