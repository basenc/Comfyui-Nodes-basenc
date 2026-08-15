from aiohttp import web
from server import PromptServer

from .env_var_node import available_env_keys
from .secret_store import store_secret


@PromptServer.instance.routes.get("/basenc/env_keys")
async def get_env_keys(request):
    return web.json_response({"keys": available_env_keys()})


@PromptServer.instance.routes.post("/basenc/api_keys")
async def save_api_key(request):
    secret = (await request.json())["secret"]
    if not isinstance(secret, str):
        raise web.HTTPBadRequest(text="`secret` must be a string.")
    return web.json_response({"reference": store_secret(secret)})
