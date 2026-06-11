from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


def greedy_decoding_kernel(
    logits: UnsafePointer[Float32, MutExternalOrigin],
    tokens: UnsafePointer[Int32, MutExternalOrigin],
    batch_size: Int32,
    vocab_size: Int32,
):
    pass


# logits, tokens are device pointers (i.e. pointers to memory on the GPU)
@export
def solve(
    logits: UnsafePointer[Float32, MutExternalOrigin],
    tokens: UnsafePointer[Int32, MutExternalOrigin],
    batch_size: Int32,
    vocab_size: Int32,
) raises:
    var threadsPerBlock: Int32 = 256
    var ctx = DeviceContext()

    var blocksPerGrid = batch_size

    var _kernel = ctx.compile_function[greedy_decoding_kernel, greedy_decoding_kernel]()
    ctx.enqueue_function(
        _kernel, logits, tokens, batch_size, vocab_size, grid_dim=blocksPerGrid, block_dim=threadsPerBlock
    )

    ctx.synchronize()
