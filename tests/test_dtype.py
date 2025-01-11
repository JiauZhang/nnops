import nnops
from nnops.tensor import Tensor
from nnops import dtype
import numpy as np

class TestDataType():
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
    def test_tensor_dtype(self):
        for tp, _ in self.types:
            tensor = Tensor(shape=[1], dtype=tp)
            assert tp == tensor.dtype

    def test_dtype_cast(self):
        for src_nps_dtype, src_np_dtype in self.types:
            for dst_nps_dtype, dst_np_dtype in self.types:
                src_np_array = (np.random.randn(5, 6, 7, 8) * 123).astype(src_np_dtype)
                src_nps_array = nnops.tensor.from_numpy(src_np_array)
                assert src_nps_array.dtype == src_nps_dtype
                dst_np_array = src_np_array.astype(dst_np_dtype)
                dst_nps_array = src_nps_array.astype(dst_nps_dtype)
                assert (dst_nps_array.numpy() == dst_np_array).all()

                # discontiguous data
                src_np_stride = src_np_array[1::2, 2::2, 3::2, 1:7:3]
                src_nps_stride = src_nps_array[1::2, 2::2, 3::2, 1:7:3]
                assert src_nps_stride.ref_count == 2
                dst_np_stride = src_np_stride.astype(dst_np_dtype)
                dst_nps_stride = src_nps_stride.astype(dst_nps_dtype)
                assert (dst_nps_stride.numpy() == dst_np_stride).all()
