import pytest
import nnops
from nnops.tensor import Tensor
from conftest import dtype_pairs as DTYPE_PAIRS, random_data
import numpy as np

@pytest.fixture(params=DTYPE_PAIRS)
def dst_dtype_pair(request):
    return request.param

class TestDataType():
    def test_tensor_dtype(self, dtype_pair, device):
        nps_type, _ = dtype_pair
        tensor = Tensor(shape=[1], dtype=nps_type).to(device)
        assert nps_type == tensor.dtype
        assert tensor.device == device

    def test_dtype_cast(self, dtype_pair, dst_dtype_pair, device):
        src_nps_dtype, src_np_dtype = dtype_pair
        dst_nps_dtype, dst_np_dtype = dst_dtype_pair
        src_np_array = random_data((5, 6, 7, 8), src_np_dtype)
        src_nps_array = nnops.tensor.from_numpy(src_np_array).to(device)
        assert src_nps_array.dtype == src_nps_dtype
        assert src_nps_array.device == device
        dst_np_array = src_np_array.astype(dst_np_dtype)
        dst_nps_array = src_nps_array.astype(dst_nps_dtype)
        src_nps_ref_count = 2 if src_nps_array.dtype == dst_nps_dtype else 1
        assert src_nps_ref_count == src_nps_array.ref_count
        assert dst_nps_array.device == device
        assert (dst_nps_array.numpy() == dst_np_array).all()

        src_np_stride = src_np_array[1::2, 2::2, 3::2, 1:7:3]
        src_nps_stride = src_nps_array[1::2, 2::2, 3::2, 1:7:3]
        assert src_nps_stride.ref_count == src_nps_ref_count + 1
        dst_np_stride = src_np_stride.astype(dst_np_dtype)
        dst_nps_stride = src_nps_stride.astype(dst_nps_dtype)
        assert dst_nps_stride.device == device
        assert (dst_nps_stride.numpy() == dst_np_stride).all()