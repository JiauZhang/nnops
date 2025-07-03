import ctypes, os
from glob import glob

IS_WINDOWS = os.name == 'nt'

def load_dll_libraries(dlls):
    for dll in dlls:
        ctypes.WinDLL(dll)

def find_cuda_dll(dll_name="cudart64_12.dll"):
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        dll_path = os.path.join(cuda_path, "bin", dll_name)
        if os.path.exists(dll_path):
            return dll_path

    for path_dir in os.environ["PATH"].split(os.pathsep):
        dll_path = os.path.join(path_dir, dll_name)
        if os.path.exists(dll_path):
            return dll_path

    return None

if IS_WINDOWS:
    if cuda_dll := find_cuda_dll():
        load_dll_libraries([cuda_dll])
    nnops_dll_path = glob(f'{os.path.join(os.path.dirname(__file__))}\\*.dll')
    load_dll_libraries(nnops_dll_path)

from .version import __version__