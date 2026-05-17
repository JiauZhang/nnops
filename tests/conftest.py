import pytest
from nnops import device, dtype
import numpy as np

cuda_available = device.CUDA.is_available()
mps_available = device.MPS.is_available()

devices = [device.CPU]
if mps_available:
    devices.append(device.MPS)

dtype_pairs = [
    (dtype.float64, np.float64),
    (dtype.float32, np.float32),
    (dtype.int64, np.int64),
    (dtype.uint64, np.uint64),
    (dtype.int32, np.int32),
    (dtype.uint32, np.uint32),
    (dtype.int16, np.int16),
    (dtype.uint16, np.uint16),
    (dtype.int8, np.int8),
    (dtype.uint8, np.uint8),
    (dtype.bool, np.bool),
]

@pytest.fixture(params=dtype_pairs)
def dtype_pair(request):
    return request.param

@pytest.fixture(params=devices)
def device(request):
    return request.param

def random_data(shape, np_dtype):
    if np.issubdtype(np_dtype, np.floating):
        return np.random.uniform(0, 100, size=shape).astype(np_dtype)
    elif np_dtype == np.bool:
        return np.random.randint(0, 2, size=shape).astype(np.bool)
    else:
        iinfo = np.iinfo(np_dtype)
        low = max(iinfo.min, 0)
        high = min(iinfo.max, 100)
        return np.random.randint(low, high + 1, size=shape).astype(np_dtype)