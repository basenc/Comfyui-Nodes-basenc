from comfy_api.latest import ComfyExtension
from typing_extensions import override

from . import api_routes
from .custom_openai_chat_completion import (
    CustomOpenAIResponse,
    ResponseInputAppend,
)
from .env_var_node import EnvVarNode
from .eval_any import Eval
from .json_path_select import JSONPathSelect
from .dimensions_preset_picker import DimensionsPresetPicker
from .rescale_to_dimensions import RescaleToDimensions
from .wan_video_size import WanVideoSize

WEB_DIRECTORY = "./web"


class CustomJsonOpenAIExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type]:
        return [
            JSONPathSelect,
            CustomOpenAIResponse,
            ResponseInputAppend,
            EnvVarNode,
            Eval,
            WanVideoSize,
            DimensionsPresetPicker,
            RescaleToDimensions,
        ]


async def comfy_entrypoint() -> CustomJsonOpenAIExtension:
    return CustomJsonOpenAIExtension()
