from aiohttp import web
from server import PromptServer

from .env_var_node import available_env_keys


@PromptServer.instance.routes.get("/basenc/env_keys")
async def get_env_keys(request):
    try:
        keys = available_env_keys()
        return web.json_response({"keys": keys})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
