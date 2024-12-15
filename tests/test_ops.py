import nnops.ops as ops
import nnops.tensor, nnops.dtype as dtype
import numpy as np

class TestOperators():
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
    def test_add(self):
        for nps_type, np_type in self.types:
            np_a = (np.random.randn(3, 1, 4) * 123).astype(np_type)
            np_b = (np.random.randn(2, 1) * 45).astype(np_type)
            np_c = (np.random.randn(5, 4) * 234).astype(np_type)
            np_d = (np.random.randn(2, 3, 5, 1) * 78).astype(np_type)
            t_a = nnops.tensor.from_numpy(np_a)
            t_b = nnops.tensor.from_numpy(np_b)
            t_c = nnops.tensor.from_numpy(np_c)
            t_d = nnops.tensor.from_numpy(np_d)
            assert (ops.add(t_a, t_b).numpy() == np_a + np_b).all()
            assert (ops.add(t_a, t_c).numpy() == np_a + np_c).all()
            assert (ops.add(t_a, t_d).numpy() == np_a + np_d).all()

    def test_add_not_contiguous(self):
        for nps_type, np_type in self.types:
            np_a = (np.random.randn(4, 5, 1, 7) * 23).astype(np_type)
            np_b = (np.random.randn(5, 5, 7) * 45).astype(np_type)
            np_a_stride = np_a[::2, ::2, :, ::3] # [2, 3, 1, 3]
            np_b_stride = np_b[::2, ::2, ::3] # [3, 2, 3]
            t_a = nnops.tensor.from_numpy(np_a_stride)
            t_b = nnops.tensor.from_numpy(np_b_stride)
            assert (ops.add(t_a, t_b).numpy() == np_a_stride + np_b_stride).all()

            np_c_stride = np_a[1::2, 2::2, :, 4:] # [2, 2, 1, 3]
            np_d_stride = np_b[2::2, 1::2, 1::2] # [2, 2, 3]
            t_c = nnops.tensor.from_numpy(np_c_stride)
            t_d = nnops.tensor.from_numpy(np_d_stride)
            assert (ops.add(t_c, t_d).numpy() == np_c_stride + np_d_stride).all()
