from typing import override

from comfy_api.latest import ComfyExtension

from . import api_routes as api_routes
from .custom_openai_chat_completion import (
    CustomOpenAICompletion,
    CustomOpenAIResponse,
    MessageAppend,
)
from .dimensions_preset_picker import DimensionsPresetPicker
from .env_var_node import EnvVarNode
from .eval_any import Eval
from .json_path_select import JSONPathSelect
from .llama_cpp_completion import LlamaCppCompletion, LlamaCppLora
from .rescale_to_dimensions import RescaleToDimensions
from .wan_video_size import WanVideoSize

WEB_DIRECTORY = "./web"


class CustomJsonOpenAIExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type]:
        return [
            JSONPathSelect,
            CustomOpenAIResponse,
            CustomOpenAICompletion,
            LlamaCppCompletion,
            LlamaCppLora,
            MessageAppend,
            EnvVarNode,
            Eval,
            WanVideoSize,
            DimensionsPresetPicker,
            RescaleToDimensions,
        ]


async def comfy_entrypoint() -> CustomJsonOpenAIExtension:
    return CustomJsonOpenAIExtension()
