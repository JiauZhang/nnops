import pytest, random
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
                if op_functor is ops.truediv:
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
                if op_functor is ops.truediv:
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
                if op_functor is ops.truediv:
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

    @pytest.mark.parametrize(
        'nnops_op, np_op', [
            (ops.add, lambda a, b: a + b),
            (ops.sub, lambda a, b: a - b),
            (ops.mul, lambda a, b: a * b),
            (ops.truediv, lambda a, b: a / b),
        ]
    )
    def test_binary_ops(self, nnops_op, np_op):
        self.cross_dtype_loop(nnops_op, np_op)

    def _np_iadd(a, b): a += b
    def _np_isub(a, b): a -= b
    def _np_imul(a, b): a *= b
    def _np_itruediv(a, b): a /= b

    @pytest.mark.parametrize(
        'nnops_op, np_op, lshape, rshape', [
            (ops.iadd, _np_iadd, (3, 4, 5), (4, 5)),
            (ops.isub, _np_isub, (6, 7, 8), (7, 1)),
            (ops.imul, _np_imul, (2, 5, 6), (1, 6)),
            (ops.itruediv, _np_itruediv, (2, 9, 3), (9, 3)),
        ]
    )
    def test_binary_ops_tensor_tensor_inplace(self, nnops_op, np_op, lshape, rshape):
        left = nnops.tensor.randn(*lshape)
        right = nnops.tensor.randn(*rshape)
        np_left = left.numpy()
        np_right = right.numpy()
        nnops_op(left, right)
        np_op(np_left, np_right)
        assert (left.numpy() == np_left).all()

    @pytest.mark.parametrize(
        'nnops_op, np_op, shape', [
            (ops.iadd, _np_iadd, (4, 4, 5)),
            (ops.isub, _np_isub, (9, 7, 8)),
            (ops.imul, _np_imul, (7, 5, 6)),
            (ops.itruediv, _np_itruediv, (5, 9, 4)),
        ]
    )
    def test_binary_ops_tensor_scalar_inplace(self, nnops_op, np_op, shape):
        left = nnops.tensor.randn(*shape)
        right = random.random()
        np_left = left.numpy()
        np_right = right
        nnops_op(left, right)
        np_op(np_left, np_right)
        assert (left.numpy() == np_left).all()

    @pytest.mark.parametrize(
        's1, s2, s3', [
            ((3, 4), (4, 6), (3, 6)),
            ((2, 1, 5, 9), (4, 3, 1, 2, 9, 7), (4, 3, 2, 2, 5, 7)),
            ((4, 1, 77, 65), (4, 3, 1, 2, 65, 66), (4, 3, 4, 2, 77, 66)),
        ]
    )
    def test_matmul_op(self, s1, s2, s3):
        n1 = np.random.randn(*s1).astype(np.float32)
        n2 = np.random.randn(*s2).astype(np.float32)
        no = n1 @ n2
        t1 = nnops.tensor.from_numpy(n1)
        t2 = nnops.tensor.from_numpy(n2)
        to = nnops.ops.matmul(t1, t2).numpy()
        assert to.dtype == no.dtype
        assert to.shape == s3
        assert (np.abs(no - to) < 1e-3).all()

        n1_stride = n1[..., ::2]
        t1_stride = t1[..., ::2]
        if len(s1) > 2:
            n2_stride = n2[..., ::2, :]
            t2_stride = t2[..., ::2, :]
        else:
            n2_stride = n2[::2, :]
            t2_stride = t2[::2, :]
        no_stride = n1_stride @ n2_stride
        to_stride = nnops.ops.matmul(t1_stride, t2_stride).numpy()
        assert to_stride.shape == s3
        assert (np.abs(no_stride - to_stride) < 1e-3).all()
