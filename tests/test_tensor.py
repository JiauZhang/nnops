from nnops.tensor import Tensor
from nnops import dtype

class TestTensor():
    def test_list_shape(self):
        shape = [1, 2, 3, 4]
        tensor = Tensor(shape=shape)
        assert shape == tensor.shape

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

    def test_tensor_reshape(self):
        runtime_error = False
        try:
            tensor_a = Tensor(shape=[1, 2, 3])
            tensor_b = tensor_a.reshape(2, 3, 1)
        except RuntimeError:
            runtime_error = True
        assert runtime_error == False
        assert tensor_a.ref_count == 2 and tensor_b.ref_count == 2

        try:
            tensor = Tensor(shape=[1, 2, 3])
            tensor.reshape(2, 3, 3)
        except RuntimeError:
            runtime_error = True
        assert runtime_error == True

    def test_tensor_ref_count_1(self):
        t_a = Tensor(shape=[2, 3, 4])
        assert t_a.ref_count == 1

        t_b = t_a[1, 1, 1]
        assert t_a.ref_count == 2 and t_a.ref_count == t_b.ref_count

        del t_a
        assert t_b.ref_count == 1

    def test_tensor_ref_count_2(self):
        t_a = Tensor(shape=[2, 3, 4])
        t_b = t_a
        assert t_a.ref_count == t_b.ref_count

        t_c = t_a[1, 1, 1]
        assert (
            t_a.ref_count == 2 and t_a.ref_count == t_b.ref_count and
            t_a.ref_count == t_c.ref_count
        )

        del t_a
        assert t_b.ref_count == 2 and t_c.ref_count == t_b.ref_count

    def test_tensor_strides(self):
        tensor = Tensor(shape=[2, 3, 4])
        stride = tensor.stride
        assert len(stride) == len(tensor.shape) and stride == [12, 4, 1]
