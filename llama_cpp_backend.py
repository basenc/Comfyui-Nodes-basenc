import asyncio
import contextlib
import gc
import json
import logging
import shlex
import socket
from stat import S_ISREG
from collections.abc import AsyncIterator
from pathlib import Path

import aiohttp
import comfy.model_management
import folder_paths
from comfy_api.latest import IO
from pydantic import TypeAdapter

from .gguf_metadata import is_lora

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


def _lora_files() -> dict[str, Path]:
    roots = {_MODELS_DIR: ""}
    for category, (folders, _) in folder_paths.folder_names_and_paths.items():
        if category not in {"custom_nodes", "datasets"}:
            for index, root in enumerate(map(Path, folders)):
                if not root.is_relative_to(_MODELS_DIR):
                    roots.setdefault(root, f"{category}/{index + 1}")
    adapters: dict[str, Path] = {}
    legacy: dict[str, str] = {}
    lora_roots = list(map(Path, folder_paths.get_folder_paths("loras")))
    for root, prefix in roots.items():
        for filename in folder_paths.recursive_search(str(root), [".git"])[0]:
            path = root / filename
            with contextlib.suppress(FileNotFoundError, PermissionError):
                stat = path.stat()
                if S_ISREG(stat.st_mode) and is_lora(path, stat.st_mtime_ns, stat.st_size):
                    name = str(Path(prefix) / filename)
                    adapters[name] = path
                    for lora_root in lora_roots:
                        if path.is_relative_to(lora_root):
                            legacy.setdefault(str(path.relative_to(lora_root)), name)
                            break
    for alias, name in legacy.items():
        if alias not in adapters:
            adapters[alias] = adapters.pop(name)
    return adapters


def _lora_inputs() -> list[IO.Combo.Input | IO.Float.Input]:
    return [
        IO.Combo.Input("lora", options=["None", *sorted(_lora_files())],
                       default="None", optional=True, tooltip="LoRA adapters identified by GGUF metadata across model folders."),
        IO.Float.Input("lora_strength", default=1.0, min=-20.0, max=20.0, step=0.05, optional=True),
    ]


_ARG_ROWS = TypeAdapter(list[tuple[str, str]])


def _llama_args_input(*rows: tuple[str, str]) -> IO.String.Input:
    field = IO.String.Input(
        "llama_server_args", default=json.dumps(rows), optional=True,
        socketless=True,
        tooltip="Appended in row order. Leave the value empty for a bare flag. "
                "The node manages --host, --port, and --parallel (-np).",
    )
    field.widget_type = "BASENC_LLAMA_ARGS"
    return field


def _custom_args(rows: str) -> list[str]:
    args: list[str] = []
    for key, value in _ARG_ROWS.validate_json(rows):
        key = key.strip()
        if not key and not value:
            continue
        if (not key.startswith("-") or key in {"-", "--"}
                or any(map(str.isspace, key)) or "=" in key):
            raise ValueError("Enter one llama-server flag as the key and its value separately.")
        args.extend((key, value) if value else (key,))
    if any(arg.partition("=")[0] in {"--host", "--port", "--parallel", "-np"} for arg in args):
        raise ValueError("The node manages llama-server --host, --port, and --parallel (-np).")
    return args


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
    draft_model: str = "None",
    spec_type: str = "draft-mtp",
    thinking: bool = True,
    reasoning_effort: str = "default",
    lora: str = "None",
    lora_strength: float = 1.0,
) -> AsyncIterator[str]:
    port = _free_port()
    args = [
        "llama-server",
        "--model", model_path,
        "--flash-attn", "on",
        *(["--ctx-size", n_ctx] if n_ctx != "auto" else []),
        "--n-gpu-layers", gpu_layers,
        "--reasoning", "on" if thinking and reasoning_effort != "none" else "off",
        *(["--reasoning-effort", reasoning_effort]
          if thinking and reasoning_effort not in {"default", "none"} else []),
        *(["--fit", "on"] if n_ctx == gpu_layers == "auto" else []),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--parallel", "1",
        "--no-ui",
        "--cors-origins", "localhost",
    ]
    if lora != "None":
        adapters = _lora_files()
        if lora not in adapters:
            raise ValueError(f"No GGUF LoRA adapter found for {lora!r}. Refresh the model list.")
        adapter = adapters[lora]
        args += ["--lora-scaled", f"{adapter}:{lora_strength}"]
    if draft_model != "None":
        args += [
            "--spec-type", spec_type,
            "--model-draft", folder_paths.get_full_path_or_raise(_MODEL_FOLDER, draft_model),
        ]
    if mmproj_path:
        args += ["--mmproj", mmproj_path]
    args.extend(_custom_args(llama_server_args))
    comfy.model_management.throw_exception_if_processing_interrupted()
    comfy.model_management.unload_all_models()
    gc.collect()
    comfy.model_management.soft_empty_cache(force=True)
    logging.info("+ %s", shlex.join(args))
    try:
        proc = await asyncio.create_subprocess_exec(*args)
    except FileNotFoundError as exc:
        raise RuntimeError("`llama-server` is not on PATH.") from exc
    async with contextlib.AsyncExitStack() as cleanup:
        cleanup.push_async_callback(_terminate, proc)
        await _wait_ready(port, proc)
        yield f"http://127.0.0.1:{port}"
