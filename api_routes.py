from typing import Literal

from aiohttp import web
from pydantic import BaseModel, Field, TypeAdapter
from server import PromptServer

from .choice_node import register_choice_outputs
from .env_var_node import available_env_keys
from .secret_store import store_secret


@PromptServer.instance.routes.get("/basenc/env_keys")
async def get_env_keys(request):
    return web.json_response({"keys": available_env_keys()})


@PromptServer.instance.routes.post("/basenc/api_keys")
async def save_api_key(request):
    body = await request.json()
    if not isinstance(body, dict) or not isinstance(body.get("secret"), str):
        raise web.HTTPBadRequest(text="`secret` must be a string.")
    if not body["secret"]:
        raise web.HTTPBadRequest(text="`secret` must not be empty.")
    return web.json_response({"reference": store_secret(body["secret"])})


class _ChoiceOutputs(BaseModel):
    node_id: Literal["TypeSafeChoice", "LlamaCppChoice"]
    count: int = Field(ge=1, le=255)


@PromptServer.instance.routes.post("/basenc/choice_outputs")
async def choice_outputs(request: web.Request) -> web.Response:
    specs = TypeAdapter(list[_ChoiceOutputs]).validate_python(await request.json())
    return web.json_response([register_choice_outputs(spec.node_id, spec.count) for spec in specs])
