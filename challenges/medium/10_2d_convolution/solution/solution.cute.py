import cutlass
import cutlass.cute as cute


@cute.kernel
def conv2d_kernel(
    input: cute.Tensor,
    kernel: cute.Tensor,
    output: cute.Tensor,
    input_rows: cute.Int32,
    input_cols: cute.Int32,
    kernel_rows: cute.Int32,
    kernel_cols: cute.Int32,
):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdx, bdy, _ = cute.arch.block_dim()

    idx = bidx * bdx + tidx
    idy = bidy * bdy + tidy

    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1

    smem_cols = bdx + kernel_cols - 1
    smem_rows = bdy + kernel_rows - 1

    smem_ptr = cute.arch.get_dyn_smem(cutlass.Float32)
    smem = cute.make_tensor(
        smem_ptr,
        cute.make_layout((smem_rows, smem_cols), stride=(smem_cols, 1)),
    )

    for i in range(tidy, smem_rows, bdy):
        input_row = bidy * bdy + i
        for j in range(tidx, smem_cols, bdx):
            input_col = bidx * bdx + j
            value = cutlass.Float32(0.0)
            if input_row < input_rows and input_col < input_cols:
                value = input[input_row, input_col]
            smem[i, j] = value
    cute.arch.sync_threads()

    sum = cutlass.Float32(0.0)
    for i in range(kernel_rows):
        for j in range(kernel_cols):
            sum += smem[tidy + i, tidx + j] * kernel[i, j]
    if idy < output_rows and idx < output_cols:
        output[idy, idx] = sum


# input, kernel, output are tensors on the GPU
@cute.jit
def solve(
    input: cute.Tensor,
    kernel: cute.Tensor,
    output: cute.Tensor,
    input_rows: cute.Int32,
    input_cols: cute.Int32,
    kernel_rows: cute.Int32,
    kernel_cols: cute.Int32,
):
    block_size = 8
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1
    input = cute.make_tensor(
        input.iterator,
        cute.make_layout((input_rows, input_cols), stride=(input_cols, 1)),
    )
    kernel = cute.make_tensor(
        kernel.iterator,
        cute.make_layout((kernel_rows, kernel_cols), stride=(kernel_cols, 1)),
    )
    output = cute.make_tensor(
        output.iterator,
        cute.make_layout((output_rows, output_cols), stride=(output_cols, 1)),
    )
    conv2d_kernel(
        input,
        kernel,
        output,
        input_rows,
        input_cols,
        kernel_rows,
        kernel_cols,
    ).launch(
        grid=[
            cute.ceil_div(output_cols, block_size),
            cute.ceil_div(output_rows, block_size),
            1,
        ],
        block=[block_size, block_size, 1],
        smem=(block_size + 30) * (block_size + 30) * cute.Float32.width // 8,
    )
