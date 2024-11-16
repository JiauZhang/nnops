from nnops.tensor import Tensor
from nnops import dtype
import numpy as np

class TestDataType():
    def test_tensor_dtype(self):
        types = [
            dtype.float32,
            dtype.int32, dtype.uint32,
            dtype.int16, dtype.uint16,
            dtype.int8, dtype.uint8,
        ]
        for tp in types:
            tensor = Tensor(shape=[1], dtype=tp)
            assert tp == tensor.dtype
