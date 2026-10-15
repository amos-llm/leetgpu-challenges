import cutlass.cute as cute


@cute.kernel
def conv1d_kernel(
    g_input: cute.Tensor,
    g_kernel: cute.Tensor,
    g_output: cute.Tensor,
    input_size: cute.Int32,
    kernel_size: cute.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    idx = bidx * bdim + tidx
    output_size = input_size - kernel_size + 1
    if idx < output_size:
        accum = 0.0
        for i in range(kernel_size):
            accum += g_input[idx + i] * g_kernel[i]
        g_output[idx] = accum


# input, kernel, output are tensors on the GPU
@cute.jit
def solve(
    input: cute.Tensor,
    kernel: cute.Tensor,
    output: cute.Tensor,
    input_size: cute.Int32,
    kernel_size: cute.Int32,
):
    output_size = input_size - kernel_size + 1

    m_input = cute.make_tensor(
        input.iterator,
        cute.make_layout((input_size,), stride=(1,)),
    )
    m_kernel = cute.make_tensor(
        kernel.iterator,
        cute.make_layout((kernel_size,), stride=(1,)),
    )
    m_output = cute.make_tensor(
        output.iterator,
        cute.make_layout((output_size,), stride=(1,)),
    )

    threads_per_block = 256
    conv1d_kernel(
        m_input,
        m_kernel,
        m_output,
        input_size,
        kernel_size,
    ).launch(
        grid=[cute.ceil_div(output_size, threads_per_block), 1, 1],
        block=[threads_per_block, 1, 1],
    )
