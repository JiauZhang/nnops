from nnops.tensor import Tensor
from nnops import dtype

class TestTensor():
    def test_list_shape(self):
        shape = [1, 2, 3, 4]
        tensor = Tensor(shape=shape)
        assert shape == tensor.shape

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

    def test_tensor_exception(self):
        runtime_error = False
        try:
            tensor = Tensor(device='fake_device')
        except RuntimeError:
            runtime_error = True
        assert runtime_error == True

    def test_tensor_nbytes_nelems(self):
        tensor = Tensor(shape=[2, 3, 4], dtype=dtype.int16)
        assert tensor.nelems == 24 and tensor.nbytes == tensor.nelems * 2

    def test_tensor_count(self):
        tensor_a = Tensor(shape=[2, 3, 4, 5])
        assert tensor_a.ref_count == 1
