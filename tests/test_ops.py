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
        [dtype.bool, np.bool],
    ]

    def cross_dtype_loop(self, op_functor, np_op_functor):
        for nps_type1, np_type1 in self.types:
            for nps_type2, np_type2 in self.types:
                # numpy boolean subtract is not supported
                if (np_type1 == np.bool or np_type2 == np.bool) and op_functor is ops.sub:
                    continue
                np_a = (np.random.randn(2, 3, 4) * 123).astype(np_type1)
                np_b = (np.random.randn(2, 3, 4) * 123).astype(np_type2)
                if op_functor is ops.div:
                    np_b[np.abs(np_b) < 1e-5] = 12.3456789
                np_ret = np_op_functor(np_a, np_b)
                nps_a = nnops.tensor.from_numpy(np_a)
                nps_b = nnops.tensor.from_numpy(np_b)
                nps_ret = op_functor(nps_a, nps_b)
                np_nps_ret = nps_ret.numpy()
                assert np_nps_ret.dtype == np_ret.dtype
                assert (np_nps_ret == np_ret).all()

                # broadcast
                np_a = (np.random.randn(3, 1, 4) * 123).astype(np_type1)
                if op_functor is ops.div:
                    np_a[np.abs(np_a) < 1e-5] = 56.789
                np_b = (np.random.randn(2, 1) * 45).astype(np_type2)
                np_c = (np.random.randn(5, 4) * 234).astype(np_type1)
                np_d = (np.random.randn(2, 3, 5, 1) * 78).astype(np_type2)
                t_a = nnops.tensor.from_numpy(np_a)
                t_b = nnops.tensor.from_numpy(np_b)
                t_c = nnops.tensor.from_numpy(np_c)
                t_d = nnops.tensor.from_numpy(np_d)
                assert (op_functor(t_b, t_a).numpy() == np_op_functor(np_b, np_a)).all()
                assert (op_functor(t_c, t_a).numpy() == np_op_functor(np_c, np_a)).all()
                assert (op_functor(t_d, t_a).numpy() == np_op_functor(np_d, np_a)).all()

                # broadcast with not contiguous tensor
                np_a = (np.random.randn(4, 5, 1, 7) * 23).astype(np_type1)
                np_b = (np.random.randn(5, 5, 7) * 45).astype(np_type2)
                if op_functor is ops.div:
                    np_b[np.abs(np_b) < 1e-5] = 98.7654321
                np_a_stride = np_a[::2, ::2, :, ::3] # [2, 3, 1, 3]
                np_b_stride = np_b[::2, ::2, ::3] # [3, 2, 3]
                t_a = nnops.tensor.from_numpy(np_a)[::2, ::2, :, ::3]
                t_b = nnops.tensor.from_numpy(np_b)[::2, ::2, ::3]
                assert t_a.is_contiguous() == False and t_b.is_contiguous() == False
                assert (op_functor(t_a, t_b).numpy() == np_op_functor(np_a_stride, np_b_stride)).all()

                np_c_stride = np_a[1::2, 2::2, :, 4:] # [2, 2, 1, 3]
                np_d_stride = np_b[2::2, 1::2, 1::2] # [2, 2, 3]
                t_c = nnops.tensor.from_numpy(np_a)[1::2, 2::2, :, 4:]
                t_d = nnops.tensor.from_numpy(np_b)[2::2, 1::2, 1::2]
                assert t_c.is_contiguous() == False and t_d.is_contiguous() == False
                assert (op_functor(t_c, t_d).numpy() == np_op_functor(np_c_stride, np_d_stride)).all()

    def test_add_op(self):
        op_functor = ops.add
        def np_op_functor(a, b):
            return a + b
        self.cross_dtype_loop(op_functor, np_op_functor)

    def test_sub_op(self):
        op_functor = ops.sub
        def np_op_functor(a, b):
            return a - b
        self.cross_dtype_loop(op_functor, np_op_functor)

    def test_mul_op(self):
        op_functor = ops.mul
        def np_op_functor(a, b):
            return a * b
        self.cross_dtype_loop(op_functor, np_op_functor)

    def test_div_op(self):
        op_functor = ops.div
        def np_op_functor(a, b):
            return a / b
        self.cross_dtype_loop(op_functor, np_op_functor)
