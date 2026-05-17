"""Tensor operations and creation functions."""

from nnops._rs import (
    Tensor,
    from_numpy,
    is_broadcastable, broadcast_shape,
    randn,
)

__all__ = [
    "Tensor",
    "from_numpy",
    "is_broadcastable",
    "broadcast_shape",
    "randn",
]