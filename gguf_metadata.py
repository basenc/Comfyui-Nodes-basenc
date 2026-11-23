import ctypes as c
import os
from contextlib import ExitStack
from functools import cache, lru_cache
from pathlib import Path


class _InitParams(c.Structure):
    _fields_ = [("no_alloc", c.c_bool), ("ctx", c.c_void_p)]


@cache
def _gguf() -> c.CDLL:
    lib = c.CDLL("libggml-base.so")
    for func, result, args in (
        (lib.gguf_init_from_file, c.c_void_p, [c.c_char_p, _InitParams]),
        (lib.gguf_free, None, [c.c_void_p]),
        (lib.gguf_find_key, c.c_int64, [c.c_void_p, c.c_char_p]),
        (lib.gguf_get_kv_type, c.c_int, [c.c_void_p, c.c_int64]),
        (lib.gguf_get_val_str, c.c_char_p, [c.c_void_p, c.c_int64]),
    ):
        func.restype, func.argtypes = result, args
    return lib


@lru_cache(maxsize=4096)
def is_lora(path: Path, mtime_ns: int, size: int) -> bool:
    with path.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            return False
    lib = _gguf()
    with ExitStack() as cleanup:
        ctx = lib.gguf_init_from_file(os.fsencode(path), _InitParams(True, None))
        if ctx:
            cleanup.callback(lib.gguf_free, ctx)
            return all(
                (index := lib.gguf_find_key(ctx, key)) >= 0
                and lib.gguf_get_kv_type(ctx, index) == 8  # GGUF_TYPE_STRING
                and lib.gguf_get_val_str(ctx, index) == value
                for key, value in ((b"general.type", b"adapter"), (b"adapter.type", b"lora"))
            )
    return False
