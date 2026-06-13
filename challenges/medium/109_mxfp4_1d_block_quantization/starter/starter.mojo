from memory import UnsafePointer


# x, y, scales are device pointers
@export
def solve(
    x: UnsafePointer[Float32],
    y: UnsafePointer[UInt8],
    scales: UnsafePointer[UInt8],
    M: Int32,
    K: Int32,
):
    pass
