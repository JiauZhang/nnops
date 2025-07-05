import os

IS_WINDOWS = os.name == 'nt'

if IS_WINDOWS:
    if cuda_path := os.environ.get("CUDA_PATH"):
        os.add_dll_directory(os.path.join(cuda_path, 'bin'))
    nnops_dll_path = os.path.join(os.path.dirname(__file__))
    os.add_dll_directory(nnops_dll_path)

from .version import __version__