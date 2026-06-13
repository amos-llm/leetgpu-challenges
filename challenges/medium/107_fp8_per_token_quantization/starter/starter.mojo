from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# x, y, scales are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutExternalOrigin],
    y: UnsafePointer[UInt8, MutExternalOrigin],
    scales: UnsafePointer[Float32, MutExternalOrigin],
    M: Int32,
    K: Int32,
) raises:
    pass
