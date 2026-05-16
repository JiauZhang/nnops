import pytest, random
import nnops.ops as ops
import nnops.tensor
from nnops import device as device_mod
from conftest import random_data
import numpy as np

class TestOperators():
    @pytest.mark.parametrize(
        'nnops_op, np_op', [
            (ops.add, lambda a, b: a + b),
            (ops.sub, lambda a, b: a - b),
            (ops.mul, lambda a, b: a * b),
            (ops.truediv, lambda a, b: a / b),
        ]
    )
    def test_binary_ops(self, nnops_op, np_op, dtype_pair, device):
        nps_type, np_type = dtype_pair
        if np_type == np.bool and nnops_op is ops.sub:
            pytest.skip("numpy boolean subtract is not supported")

        np_a = random_data((2, 3, 4), np_type)
        np_b = random_data((2, 3, 4), np_type)
        if nnops_op is ops.truediv:
            np_b[np.abs(np_b) < 1e-5] = 12.3456789
        np_ret = np_op(np_a, np_b)
        nps_a = nnops.tensor.from_numpy(np_a).to(device)
        nps_b = nnops.tensor.from_numpy(np_b).to(device)
        nps_ret = nnops_op(nps_a, nps_b).to(device_mod.CPU)
        np_nps_ret = nps_ret.numpy()
        assert np_nps_ret.dtype == np_ret.dtype
        assert (np_nps_ret == np_ret).all()

        np_a = random_data((3, 1, 4), np_type)
        if nnops_op is ops.truediv:
            np_a[np.abs(np_a) < 1e-5] = 56.789
        np_b = random_data((2, 1), np_type)
        np_c = random_data((5, 4), np_type)
        np_d = random_data((2, 3, 5, 1), np_type)
        t_a = nnops.tensor.from_numpy(np_a).to(device)
        t_b = nnops.tensor.from_numpy(np_b).to(device)
        t_c = nnops.tensor.from_numpy(np_c).to(device)
        t_d = nnops.tensor.from_numpy(np_d).to(device)
        assert (nnops_op(t_b, t_a).to(device_mod.CPU).numpy() == np_op(np_b, np_a)).all()
        assert (nnops_op(t_c, t_a).to(device_mod.CPU).numpy() == np_op(np_c, np_a)).all()
        assert (nnops_op(t_d, t_a).to(device_mod.CPU).numpy() == np_op(np_d, np_a)).all()

        np_a = random_data((4, 5, 1, 7), np_type)
        np_b = random_data((5, 5, 7), np_type)
        if nnops_op is ops.truediv:
            np_b[np.abs(np_b) < 1e-5] = 98.7654321
        np_a_stride = np_a[::2, ::2, :, ::3]
        np_b_stride = np_b[::2, ::2, ::3]
        t_a = nnops.tensor.from_numpy(np_a).to(device)[::2, ::2, :, ::3]
        t_b = nnops.tensor.from_numpy(np_b).to(device)[::2, ::2, ::3]
        assert t_a.is_contiguous() == False and t_b.is_contiguous() == False
        assert (nnops_op(t_a, t_b).to(device_mod.CPU).numpy() == np_op(np_a_stride, np_b_stride)).all()

        np_c_stride = np_a[1::2, 2::2, :, 4:]
        np_d_stride = np_b[2::2, 1::2, 1::2]
        t_c = nnops.tensor.from_numpy(np_a).to(device)[1::2, 2::2, :, 4:]
        t_d = nnops.tensor.from_numpy(np_b).to(device)[2::2, 1::2, 1::2]
        assert t_c.is_contiguous() == False and t_d.is_contiguous() == False
        assert (nnops_op(t_c, t_d).to(device_mod.CPU).numpy() == np_op(np_c_stride, np_d_stride)).all()

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
    def test_binary_ops_tensor_tensor_inplace(self, nnops_op, np_op, lshape, rshape, device):
        left = nnops.tensor.randn(*lshape).to(device)
        right = nnops.tensor.randn(*rshape).to(device)
        np_left = left.to(device_mod.CPU).numpy()
        np_right = right.to(device_mod.CPU).numpy()
        nnops_op(left, right)
        np_op(np_left, np_right)
        assert (left.to(device_mod.CPU).numpy() == np_left).all()

    @pytest.mark.parametrize(
        'nnops_op, np_op, shape', [
            (ops.iadd, _np_iadd, (4, 4, 5)),
            (ops.isub, _np_isub, (9, 7, 8)),
            (ops.imul, _np_imul, (7, 5, 6)),
            (ops.itruediv, _np_itruediv, (5, 9, 4)),
        ]
    )
    def test_binary_ops_tensor_scalar_inplace(self, nnops_op, np_op, shape, device):
        left = nnops.tensor.randn(*shape).to(device)
        right = random.random()
        np_left = left.to(device_mod.CPU).numpy()
        np_right = right
        nnops_op(left, right)
        np_op(np_left, np_right)
        assert (left.to(device_mod.CPU).numpy() == np_left).all()

    @pytest.mark.parametrize(
        's1, s2, s3', [
            ((3, 4), (4, 6), (3, 6)),
            ((2, 1, 5, 9), (4, 3, 1, 2, 9, 7), (4, 3, 2, 2, 5, 7)),
            ((4, 1, 77, 65), (4, 3, 1, 2, 65, 66), (4, 3, 4, 2, 77, 66)),
        ]
    )
    def test_matmul_op(self, s1, s2, s3, device):
        n1 = np.random.randn(*s1).astype(np.float32)
        n2 = np.random.randn(*s2).astype(np.float32)
        no = n1 @ n2
        t1 = nnops.tensor.from_numpy(n1).to(device)
        t2 = nnops.tensor.from_numpy(n2).to(device)
        to = nnops.ops.matmul(t1, t2).to(device_mod.CPU).numpy()
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
        to_stride = nnops.ops.matmul(t1_stride, t2_stride).to(device_mod.CPU).numpy()
        assert to_stride.shape == s3
        assert (np.abs(no_stride - to_stride) < 1e-3).all()

class TestLinear():
    def test_linear_without_bias(self, device):
        batch, in_features, out_features = 4, 8, 16
        input_np = np.random.randn(batch, in_features).astype(np.float32)
        weight_np = np.random.randn(out_features, in_features).astype(np.float32)
        expected = input_np @ weight_np.T

        input_t = nnops.tensor.from_numpy(input_np).to(device)
        weight_t = nnops.tensor.from_numpy(weight_np).to(device)
        result = nnops.ops.linear(input_t, weight_t).to(device_mod.CPU).numpy()
        assert result.dtype == np.float32
        assert result.shape == (batch, out_features)
        assert (np.abs(result - expected) < 1e-3).all()

    def test_linear_with_bias(self, device):
        batch, in_features, out_features = 3, 10, 5
        input_np = np.random.randn(batch, in_features).astype(np.float32)
        weight_np = np.random.randn(out_features, in_features).astype(np.float32)
        bias_np = np.random.randn(out_features).astype(np.float32)
        expected = input_np @ weight_np.T + bias_np

        input_t = nnops.tensor.from_numpy(input_np).to(device)
        weight_t = nnops.tensor.from_numpy(weight_np).to(device)
        bias_t = nnops.tensor.from_numpy(bias_np).to(device)
        result = nnops.ops.linear(input_t, weight_t, bias_t).to(device_mod.CPU).numpy()
        assert result.dtype == np.float32
        assert result.shape == (batch, out_features)
        assert (np.abs(result - expected) < 1e-3).all()