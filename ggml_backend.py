import ctypes
from importlib.metadata import distribution

_lib_dir = distribution("llama-cpp-python").locate_file("llama_cpp/lib")
ctypes.CDLL(str(_lib_dir / "libggml-base.so"), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(str(_lib_dir / "libggml.so"), mode=ctypes.RTLD_GLOBAL)
