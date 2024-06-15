from nnops.tensor import Tensor, TensorShape
from nnops import dtype

class TestTensor():
    def test_list_shape(self):
        shape = [1, 2, 3, 4]
        tensor = Tensor(shape=shape)
        assert shape == tensor.shape

    def test_tensor_shape(self):
        shape = TensorShape(shape=[1, 2, 3, 4])
        tensor = Tensor(shape=shape)
        assert shape.get_dims() == tensor.shape

    def test_tensor_dtype(self):
        types = [
            dtype.float32, dtype.float16,
            dtype.int32, dtype.uint32,
            dtype.int16, dtype.uint16,
            dtype.int8, dtype.uint8,
        ]
        for tp in types:
            tensor = Tensor(dtype=tp)
            assert tp == tensor.dtype
