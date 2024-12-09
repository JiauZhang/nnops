import nnops
from nnops.tensor import Tensor
from nnops import dtype
import numpy as np
import random

class TestTensor():
    types = [
        [dtype.float64, np.float64],
        [dtype.float32, np.float32],
        [dtype.int64, np.int64], 
        [dtype.uint64, np.uint64],
        [dtype.int32, np.int32], 
        [dtype.uint32, np.uint32],
        [dtype.int16, np.int16],
        [dtype.uint16, np.uint16],
        [dtype.int8, np.int8],
        [dtype.uint8, np.uint8],
    ]

    def test_list_shape(self):
        shape = [1, 2, 3, 4]
        tensor = Tensor(shape=shape)
        assert shape == tensor.shape

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

    def test_tensor_reshape_1(self):
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

    def test_tensor_reshape_2(self):
        t_a = Tensor(shape=[4, 5, 6])
        t_b = t_a.reshape(5, 6, 4)
        t_c = t_b.reshape(-1)
        assert t_a.ref_count == 3 and t_c.ref_count == t_a.ref_count

        t_d = t_a[::2, :, ::3] # shape: [2, 5, 2]
        t_e = t_d.reshape(10, 2)
        assert t_a.ref_count == 4 and t_e.ref_count == 1

    def test_tensor_reshape_3(self):
        np_t = np.random.randn(6, 7, 8).astype(np.float32)
        np_t_stride = np_t[1::2, 2::3, 3::2] # [3, 2, 3]
        np_t_reshape = np_t_stride.reshape(2, 9)
        nop_t = nnops.tensor.from_numpy(np_t)
        nop_t_stride = nop_t[1::2, 2::3, 3::2]
        nop_t_reshape = nop_t_stride.reshape(2, 9)
        assert (nop_t_reshape.numpy() == np_t_reshape).all()

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

    def test_tensor_slice_1(self):
        t_a = Tensor(shape=[2, 2, 3])
        t_b = t_a[0]
        assert t_a.ref_count == 2 and t_a.ref_count == t_b.ref_count
        assert t_b.shape == t_a.shape[1:]
        assert t_b.stride == t_a.stride[1:]
        assert t_b.nelems == 6

        t_c = t_a[1, 1]
        assert t_a.ref_count == 3 and t_a.ref_count == t_c.ref_count
        assert t_c.shape == t_a.shape[-1:]
        assert t_c.stride == t_a.stride[-1:]
        assert t_c.nelems == 3

        t_d = t_a[1, 1, 2]
        assert t_d.nelems == 1 and t_d.ndim == 0

    def test_tensor_slice_2(self):
        t_a = Tensor(shape=[3, 4, 5])
        t_b = t_a[0]
        assert type(t_b) == type(t_a) and isinstance(t_b, Tensor)
        assert t_b.shape == t_a.shape[1:] and t_b.ref_count == 2 and t_b.ref_count == t_a.ref_count

        t_c = t_b[0]
        assert type(t_b) == type(t_c) and isinstance(t_c, Tensor) and t_c.shape == t_a.shape[2:]
        assert t_c.ref_count == 3 and t_b.ref_count == t_a.ref_count and t_b.ref_count == t_c.ref_count

    def test_tensor_slice_3(self):
        t_a = Tensor(shape=[4, 5, 6])
        t_b = t_a[::2, 1::3, 2:5:]
        assert t_b.shape == [2, 2, 3] and t_b.nelems == 12

        t_c = t_b[:, :, ::2]
        assert t_c.shape == [2, 2, 2] and t_c.nelems == 8

        t_d = t_a[:, 0, :]
        assert t_d.nelems == 24 and t_d.ref_count == 4

        t_e = t_a[..., 0]
        t_f = t_a[0, ...]
        assert t_e.shape == [4, 5] and t_f.shape == [5, 6]

    def test_tensor_slice_4(self):
        t_a = Tensor(shape=[3, 4, 5, 6, 7])
        t_b = t_a[1, ..., 3]
        assert t_b.shape == [4, 5, 6]

        t_c = t_a[1, 2, ...]
        assert t_c.shape == [5, 6, 7]

        t_d = t_a[-1, 3, ..., 5, 6]
        assert t_d.shape == [5]

        t_e = t_a[-1, ::2, ..., 5, 6]
        assert t_e.shape == [2, 5]

    def test_tensor_contiguous(self):
        t_a = Tensor(shape=[4, 5, 9])
        t_b = t_a[::2, ::2]
        assert t_a.is_contiguous() == True and t_b.is_contiguous() == False

        t_c = t_b[1, 1]
        t_d = t_a[..., ::2]
        assert t_c.is_contiguous() == True and t_d.is_contiguous() == False

    def test_tensor_numpy(self):
        for nnops_type, np_type in self.types:
            np_a = (np.random.randn(4, 5, 6) * 9876).astype(np_type)
            nnops_a = nnops.tensor.from_numpy(np_a)
            nnops_np = nnops_a.numpy()
            assert (np_a == nnops_np).all() and nnops_a.dtype == nnops_type
            assert nnops_a.ref_count == 1 and nnops_a.shape == [4, 5, 6]

            t_b = nnops_a[::2, ::2, ::3]
            assert t_b.ref_count == 2

            del nnops_a
            assert t_b.ref_count == 1

    def test_tensor_broadcast(self):
        t_a = Tensor(shape=[3, 1, 4])
        t_b = Tensor(shape=[3, 1])
        assert nnops.tensor.is_broadcastable(t_a, t_b) == True
        assert nnops.tensor.broadcast_shape(t_b, t_a) == [3, 3, 4]

        t_c = t_a.broadcast_to((2, 3, 2, 4))
        assert t_c.shape == [2, 3, 2, 4] and t_c.stride == [0, 4, 0, 1]

        np_tensor = np.random.randn(3, 1, 4).astype(np.float32)
        t_d = nnops.tensor.from_numpy(np_tensor)
        np_broadcast = np.broadcast_to(np_tensor, (2, 3, 5, 4))
        t_d_broadcast = t_d.broadcast_to((2, 3, 5, 4))
        assert t_d_broadcast.shape == [2, 3, 5, 4]
        assert t_d_broadcast.stride == [0, 4, 0, 1]
        assert (np_broadcast == t_d_broadcast.numpy()).all()

    def test_tensor_contiguous(self):
        t_a = nnops.tensor.from_numpy(np.random.randn(4, 5, 6))
        t_b = t_a[::2, ::2, ::2]
        assert t_b.is_contiguous() == False

        t_c = t_b.contiguous()
        assert t_c.is_contiguous() == True
        assert t_c.ref_count == 1 and t_b.ref_count == 2

        t_d = t_b.clone()
        assert t_d.is_contiguous() == True
        assert t_d.ref_count == 1 and t_b.ref_count == 2
        assert (t_d.numpy() == t_b.numpy()).all()

    def test_tensor_permute(self):
        np_a = np.random.randn(4, 5, 6, 7)
        nnops_a = nnops.tensor.from_numpy(np_a)
        permute_index = [0, 3, 1, 2]
        random.shuffle(permute_index)
        nnops_permute = nnops_a.permute(*tuple(permute_index))
        np_permute = np.transpose(np_a, permute_index)
        assert nnops_permute.ref_count == 2
        assert tuple(nnops_permute.shape) == np_permute.shape
        assert (nnops_permute.numpy() == np_permute).all()

        np_b = np_a[1::2, 1::2, 1::2, 2::3]
        nnops_b = nnops_a[1::2, 1::2, 1::2, 2::3]
        random.shuffle(permute_index)
        nnops_permutex2 = nnops_b.permute(*tuple(permute_index))
        np_permutex2 = np.transpose(np_b, permute_index)
        assert nnops_permutex2.ref_count == 4
        assert tuple(nnops_permutex2.shape) == np_permutex2.shape
        assert (nnops_permutex2.numpy() == np_permutex2).all()