from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# Q, K, V, parents, output are device pointers
@export
def solve(
    Q: UnsafePointer[Float32, MutExternalOrigin],
    K: UnsafePointer[Float32, MutExternalOrigin],
    V: UnsafePointer[Float32, MutExternalOrigin],
    parents: UnsafePointer[Int32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    T: Int32,
    D: Int32,
) raises:
    pass
