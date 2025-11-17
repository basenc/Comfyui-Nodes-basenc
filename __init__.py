from comfy_api.latest import ComfyExtension
from typing_extensions import override

from .custom_openai_chat_completion import ChatMessagesCreate, CustomOpenAIChatCompletion
from .env_var_node import EnvVarNode
from .eval_any import Eval
from .json_path_select import JSONPathSelect
from .wan_video_size import WanVideoSize


class CustomJsonOpenAIExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type]:
        return [
            JSONPathSelect,
            CustomOpenAIChatCompletion,
            ChatMessagesCreate,
            EnvVarNode,
            Eval,
            WanVideoSize,
        ]


async def comfy_entrypoint() -> CustomJsonOpenAIExtension:
    return CustomJsonOpenAIExtension()
