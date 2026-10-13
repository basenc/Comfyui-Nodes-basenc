import asyncio
import contextlib
import gc
import shlex
import socket
from collections.abc import AsyncIterator
from pathlib import Path

import aiohttp
import comfy.model_management
import folder_paths

_MODEL_FOLDER = "basenc_llama_cpp_models"
_MODELS_DIR = Path(folder_paths.models_dir)


def _register_model_folders(name: str, extensions: set[str], *folders: Path) -> None:
    for index, folder in enumerate(folders):
        folder_paths.add_model_folder_path(name, str(folder), is_default=index == 0)
    folder_paths.folder_names_and_paths[name] = (
        folder_paths.folder_names_and_paths[name][0],
        extensions,
    )


_register_model_folders(
    _MODEL_FOLDER,
    {".gguf"},
    _MODELS_DIR / "LLM",
    _MODELS_DIR / "text_encoders",
)


def _model_options(mmproj: bool) -> list[str]:
    return [
        name
        for name in folder_paths.get_filename_list(_MODEL_FOLDER)
        if ("mmproj" in Path(name).name.casefold()) == mmproj
    ]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _wait_ready(port: int, proc: asyncio.subprocess.Process) -> None:
    url = f"http://127.0.0.1:{port}/health"
    async with aiohttp.ClientSession() as session:
        while True:
            comfy.model_management.throw_exception_if_processing_interrupted()
            if proc.returncode is not None:
                raise RuntimeError(f"llama-server exited with code {proc.returncode}")
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=1.0)) as response:
                    body = await response.json()
                    if response.status == 200 and body.get("status") == "ok":
                        return
            except (aiohttp.ClientError, TimeoutError, ValueError):
                pass
            await asyncio.sleep(0.2)


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=10)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()


@contextlib.asynccontextmanager
async def _llama_server(
    *,
    model_path: str,
    mmproj_path: str | None,
    n_ctx: str,
    gpu_layers: str,
    llama_server_args: str,
    mtp_model: str = "None",
) -> AsyncIterator[str]:
    port = _free_port()
    args = [
        "llama-server",
        *shlex.split(llama_server_args),
        "--model", model_path,
        *(["--ctx-size", n_ctx] if n_ctx != "auto" else []),
        "--n-gpu-layers", gpu_layers,
        "--fit", "on",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--no-webui",
    ]
    if mtp_model != "None":
        args += [
            "--spec-type", "draft-mtp",
            "--model-draft", folder_paths.get_full_path_or_raise(_MODEL_FOLDER, mtp_model),
        ]
    if mmproj_path:
        args += ["--mmproj", mmproj_path]
    comfy.model_management.throw_exception_if_processing_interrupted()
    comfy.model_management.unload_all_models()
    gc.collect()
    comfy.model_management.soft_empty_cache(force=True)
    try:
        proc = await asyncio.create_subprocess_exec(*args)
    except FileNotFoundError as exc:
        raise RuntimeError("`llama-server` is not on PATH.") from exc
    async with contextlib.AsyncExitStack() as cleanup:
        cleanup.push_async_callback(_terminate, proc)
        await _wait_ready(port, proc)
        yield f"http://127.0.0.1:{port}"
