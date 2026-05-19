"""High-performance neural network operator library."""

from nnops._rs import __version__

from nnops._rs import Tensor, from_numpy, randn
from nnops._rs import is_broadcastable, broadcast_shape
from nnops._rs import add, sub, mul, truediv
from nnops._rs import iadd, isub, imul, itruediv
from nnops._rs import matmul, linear
from nnops._rs import DeviceType, DataType
from nnops._rs import CPU, CUDA, MPS
from nnops._rs import float64, float32, int64, uint64, int32, uint32
from nnops._rs import int16, uint16, int8, uint8, bool
from nnops._rs import is_device_available, show_device_info

from nnops import tensor
from nnops import ops
from nnops import device
from nnops import dtype

__all__ = [
    "__version__",
    "Tensor", "from_numpy", "randn",
    "is_broadcastable", "broadcast_shape",
    "add", "sub", "mul", "truediv",
    "iadd", "isub", "imul", "itruediv",
    "matmul", "linear",
    "DeviceType", "DataType",
    "CPU", "CUDA", "MPS",
    "float64", "float32", "int64", "uint64", "int32", "uint32",
    "int16", "uint16", "int8", "uint8", "bool",
    "is_device_available", "show_device_info",
    "tensor", "ops", "device", "dtype",
]