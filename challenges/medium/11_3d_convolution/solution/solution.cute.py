import cutlass
import cutlass.cute as cute


@cute.kernel
def conv3d_kernel(
    input: cute.Tensor,
    kernel: cute.Tensor,
    output: cute.Tensor,
    input_depth: cute.Int32,
    input_rows: cute.Int32,
    input_cols: cute.Int32,
    kernel_depth: cute.Int32,
    kernel_rows: cute.Int32,
    kernel_cols: cute.Int32,
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdx, bdy, bdz = cute.arch.block_dim()

    idx = bidx * bdx + tidx
    idy = bidy * bdy + tidy
    idz = bidz * bdz + tidz

    output_depth = input_depth - kernel_depth + 1
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1

    smem_cols = bdx + kernel_cols - 1
    smem_rows = bdy + kernel_rows - 1
    smem_depth = bdz + kernel_depth - 1

    smem_ptr = cute.arch.get_dyn_smem(cutlass.Float32)
    smem = cute.make_tensor(
        smem_ptr,
        cute.make_layout(
            (smem_depth, smem_rows, smem_cols),
            stride=(smem_rows * smem_cols, smem_cols, 1),
        ),
    )

    for i in range(tidz, smem_depth, bdz):
        in_depth = bidz * bdz + i
        for j in range(tidy, smem_rows, bdy):
            in_row = bidy * bdy + j
            for k in range(tidx, smem_cols, bdx):
                in_col = bidx * bdx + k
                value = cutlass.Float32(0.0)
                if in_depth < input_depth and in_row < input_rows and in_col < input_cols:
                    value = input[in_depth, in_row, in_col]
                smem[i, j, k] = value
    cute.arch.sync_threads()

    sum = cutlass.Float32(0.0)
    for i in range(kernel_depth):
        for j in range(kernel_rows):
            for k in range(kernel_cols):
                sum += smem[tidz + i, tidy + j, tidx + k] * kernel[i, j, k]
    if idz < output_depth and idy < output_rows and idx < output_cols:
        output[idz, idy, idx] = sum


# input, kernel, output are tensors on the GPU
@cute.jit
def solve(
    input: cute.Tensor,
    kernel: cute.Tensor,
    output: cute.Tensor,
    input_depth: cute.Int32,
    input_rows: cute.Int32,
    input_cols: cute.Int32,
    kernel_depth: cute.Int32,
    kernel_rows: cute.Int32,
    kernel_cols: cute.Int32,
):
    block_size = 8
    output_depth = input_depth - kernel_depth + 1
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1
    input = cute.make_tensor(
        input.iterator,
        cute.make_layout(
            (input_depth, input_rows, input_depth),
            stride=(input_rows * input_cols, input_cols, 1),
        ),
    )
    kernel = cute.make_tensor(
        kernel.iterator,
        cute.make_layout(
            (kernel_depth, kernel_rows, kernel_cols),
            stride=(kernel_rows * kernel_cols, kernel_cols, 1),
        ),
    )
    output = cute.make_tensor(
        output.iterator,
        cute.make_layout(
            (output_depth, output_rows, output_depth),
            stride=(output_rows * output_cols, output_cols, 1),
        ),
    )
    conv3d_kernel(
        input,
        kernel,
        output,
        input_depth,
        input_rows,
        input_cols,
        kernel_depth,
        kernel_rows,
        kernel_cols,
    ).launch(
        grid=[
            cute.ceil_div(output_cols, block_size),
            cute.ceil_div(output_rows, block_size),
            cute.ceil_div(output_depth, block_size),
        ],
        block=[block_size, block_size, block_size],
        smem=(block_size + 4) * (block_size + 4) * (block_size + 4) * cute.Float32.width // 8,
    )
