from std.gpu.host import DeviceContext
from std.memory import UnsafePointer


# Q, K_cache, V_cache, block_table, cu_seqlens, output are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutExternalOrigin],
    K_cache: UnsafePointer[Float32, MutExternalOrigin],
    V_cache: UnsafePointer[Float32, MutExternalOrigin],
    block_table: UnsafePointer[Int32, MutExternalOrigin],
    cu_seqlens: UnsafePointer[Int32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    T: Int32,
    num_heads: Int32,
    head_dim: Int32,
    block_size: Int32,
    max_blocks_per_seq: Int32,
    S: Int32,
) raises:
    pass
