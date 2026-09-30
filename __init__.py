from typing import override

from comfy_api.latest import ComfyExtension

from . import ggml_backend as ggml_backend
from . import api_routes as api_routes
from .env_var_node import EnvVarNode
from .eval_node import Eval
from .image_dimensions_node import ImageDimensions
from .llama_cpp_completion_node import LlamaCppCompletion
from .message_node import AddMessage
from .openai_chat_node import OpenAIChatCompletion, OpenAIResponse


WEB_DIRECTORY = "./web"


class CustomJsonOpenAIExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type]:
        return [
            OpenAIResponse,
            OpenAIChatCompletion,
            LlamaCppCompletion,
            AddMessage,
            EnvVarNode,
            Eval,
            ImageDimensions,
        ]


async def comfy_entrypoint() -> CustomJsonOpenAIExtension:
    return CustomJsonOpenAIExtension()
