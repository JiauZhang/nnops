"""Device management utilities."""

from nnops._rs import (
    CPU,
    CUDA,
    NPU,
    MPS,
    show_device_info,
    is_device_available,
)

__all__ = [
    "CPU",
    "CUDA",
    "NPU",
    "MPS",
    "show_device_info",
    "is_device_available",
]