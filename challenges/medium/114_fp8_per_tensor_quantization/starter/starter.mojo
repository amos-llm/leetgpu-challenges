from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import UnsafePointer
from std.math import ceildiv


# x, y, scale_out are device pointers
@export
def solve(
    x: UnsafePointer[Float32, MutExternalOrigin],
    y: UnsafePointer[UInt8, MutExternalOrigin],
    scale_out: UnsafePointer[Float32, MutExternalOrigin],
    N: Int32,
) raises:
    pass
