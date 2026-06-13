from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# Q, K_new, V_new, K_cache, V_cache, output are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutExternalOrigin],
    K_new: UnsafePointer[Float32, MutExternalOrigin],
    V_new: UnsafePointer[Float32, MutExternalOrigin],
    K_cache: UnsafePointer[Float32, MutExternalOrigin],
    V_cache: UnsafePointer[Float32, MutExternalOrigin],
    seq_len: Int32,
    output: UnsafePointer[Float32, MutExternalOrigin],
    H: Int32,
    D: Int32,
) raises:
    pass
