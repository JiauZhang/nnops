"""Operator functions for tensor operations."""

from nnops._rs import (
    add, sub, mul, truediv,
    iadd, isub, imul, itruediv,
    matmul, linear,
)

__all__ = [
    "add", "sub", "mul", "truediv",
    "iadd", "isub", "imul", "itruediv",
    "matmul", "linear",
]