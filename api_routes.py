import os
from pathlib import Path
from typing import Optional

from aiohttp import web
from dotenv import dotenv_values
from server import PromptServer


def _node_root() -> Path:
    return Path(__file__).resolve().parent


def _dotenv_path() -> Optional[Path]:
    comfy_root = _node_root().parent.parent
    env_file = comfy_root / ".env"
    return env_file if env_file.exists() else None


def _dotenv_keys() -> list[str]:
    dotenv_path = _dotenv_path()
    if not dotenv_path:
        return []
    return list(dotenv_values(dotenv_path).keys())


def _available_env_keys() -> list[str]:
    keys: set[str] = set(_dotenv_keys())

    keys.update(os.environ.keys())

    return sorted(keys)


@PromptServer.instance.routes.get("/basenc/env_keys")
async def get_env_keys(request):
    try:
        keys = _available_env_keys()
        return web.json_response({"keys": keys})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
