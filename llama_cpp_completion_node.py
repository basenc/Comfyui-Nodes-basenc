import ctypes
import gc
import json
from functools import cache
from pathlib import Path
from typing import TypedDict

import ai
import comfy.model_management
import folder_paths
from ai.types import media
from ai.types.usage import Usage
from comfy_api.latest import IO
from llama_cpp import Llama, llama_cpp
from llama_cpp.llama_chat_format import MTMDChatHandler

from .message_node import MESSAGES, Messages
from .openai_chat_node import JsonValue, _parse_tools


type ChatMessage = dict[str, JsonValue]
type ChatTool = dict[str, JsonValue]


class CompletionRequest(TypedDict):
    model_path: str
    mmproj_path: str | None
    messages: list[ChatMessage]
    tools: list[ChatTool]
    tool_choice: str | None
    thinking: bool
    chat_template_kwargs: dict[str, JsonValue]
    temperature: float
    max_tokens: int | None
    n_ctx: int
    n_batch: int
    gpu_layers: int
    verbose: boolODEL_FOLDER = "basenc_llama_cpp_models"
_MODELS_DIR = Path(folder_paths.models_dir)


def _register_model_folders(
    name: str,
    extensions: set[str],
    *folders: Path,
) -> None:
    for index, folder in enumerate(folders):
        folder_paths.add_model_folder_path(
            name,
            str(folder),
            is_default=index == 0,
        )
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


def _run_completion(request: CompletionRequest) -> dict[str, JsonValue]:
    comfy.model_management.throw_exception_if_processing_interrupted()
    comfy.model_management.unload_all_models()
    gc.collect()
    comfy.model_management.soft_empty_cache(force=True)
    with subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("llama_cpp_worker.py"))],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    ) as worker:
        payload: bytes | None = json.dumps(request).encode()
        try:
            while True:
                comfy.model_management.throw_exception_if_processing_interrupted()
                try:
                    output, _ = worker.communicate(input=payload, timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    payload = None
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.communicate()
        if worker.returncode:
            raise RuntimeError(f"llama.cpp worker exited with code {worker.returncode}")
    return json.loads(output)


def _tool_result_content(value: JsonValue | ai.messages.ContentOutput) -> str:
    """Convert an AI SDK tool result to one OpenAI tool-content string."""
    if isinstance(value, ai.messages.ContentOutput):
        raise TypeError("MTMD does not support multipart tool results.")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), default=str)


def _messages_to_openai_chat(messages: Messages) -> list[ChatMessage]:
    """Bridge AI SDK messages to the OpenAI dictionaries required by llama-cpp-python."""
    result: list[ChatMessage] = []
    for message in messages:
        if message.role == "system":
            result.append({"role": "system", "content": message.text})
        elif message.role == "user":
            if message.files:
                content: list[dict[str, JsonValue]] = []
                for part in message.parts:
                    if isinstance(part, ai.messages.TextPart):
                        content.append({"type": "text", "text": part.text})
                    elif isinstance(part, ai.messages.FilePart):
                        if not part.media_type.startswith(
                            ("image/", "audio/", "video/")
                        ):
                            raise ValueError(
                                "MTMD messages support image, audio, and video file parts only."
                            )
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": media.data_to_data_url(
                                        part.data, part.media_type
                                    )
                                },
                            }
                        )
                result.append({"role": "user", "content": content})
            else:
                result.append({"role": "user", "content": message.text})
        elif message.role == "assistant":
            assistant: ChatMessage = {"role": "assistant"}
            if message.text:
                assistant["content"] = message.text
            if message.tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": call.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": call.tool_name,
                            "arguments": call.tool_args,
                        },
                    }
                    for call in message.tool_calls
                ]
            result.append(assistant)
        elif message.role == "tool":
            result.extend(
                {
                    "role": "tool",
                    "tool_call_id": part.tool_call_id,
                    "content": _tool_result_content(part.get_model_input()),
                }
                for part in message.tool_results
            )
    return result


def _tools_to_openai_chat(tools: list[ai.tools.Tool]) -> list[ChatTool]:
    """Convert AI SDK tool schemas to llama-cpp-python's function tools."""
    result: list[ChatTool] = []
    for tool in tools:
        if tool.kind != "function" or tool.spec is None:
            raise ValueError("Local chat generation supports function tools only.")
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.spec.description or "",
                    "parameters": tool.spec.params,
                },
            }
        )
    return result


def _assistant_message(
    response: dict[str, JsonValue],
) -> ai.messages.Message:
    """Bridge a llama-cpp-python response back to an AI SDK message."""
    raw = response["choices"][0]["message"]
    parts: list[ai.messages.Part] = []
    if raw.get("content"):
        parts.append(ai.messages.TextPart(id="text", text=raw["content"]))
    parts.extend(
        ai.messages.ToolCallPart(
            id=call["id"],
            tool_call_id=call["id"],
            tool_name=call["function"]["name"],
            tool_args=call["function"]["arguments"],
        )
        for call in raw.get("tool_calls", [])
    )
    usage = response["usage"]
    return ai.messages.Message(
        role="assistant",
        parts=parts,
        usage=Usage(
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            raw=dict(usage),
        ),
    )


class LlamaCppCompletion(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="LlamaCppCompletion",
            display_name="llama.cpp Chat Completion",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=_model_options(mmproj=False),
                    tooltip="Text-model GGUF from models/LLM or models/text_encoders.",
                ),
                IO.Combo.Input(
                    "mmproj",
                    options=["None", *_model_options(mmproj=True)],
                    default="None",
                    tooltip="Optional MTMD projector. None runs text-only chat.",
                ),
                MESSAGES.Input(
                    "messages",
                    optional=False,
                ),
                IO.String.Input(
                    "tools_json",
                    default="[]",
                    multiline=True,
                    optional=True,
                    socketless=False,
                    force_input=False,
                ),
                IO.Combo.Input(
                    "tool_choice",
                    options=["auto", "none", "required"],
                    default="auto",
                    optional=True,
                    tooltip="Tool choice policy. Applies when tools are provided.",
                ),
                IO.Boolean.Input(
                    "thinking",
                    default=True,
                    optional=True,
                    tooltip="Pass enable_thinking to the model chat template.",
                ),
                IO.String.Input(
                    "chat_template_kwargs",
                    default="{}",
                    multiline=True,
                    optional=True,
                    socketless=False,
                    force_input=False,
                    tooltip="Additional GGUF chat-template variables as a JSON object.",
                ),
                IO.Float.Input(
                    "temperature",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    optional=True,
                    tooltip="Sampling temperature.",
                ),
                IO.Int.Input(
                    "max_output_tokens",
                    default=1024,
                    min=0,
                    max=2_147_483_647,
                    optional=True,
                    tooltip="Max output tokens. 0 to use the remaining context.",
                ),
                IO.Int.Input(
                    "n_ctx",
                    default=8192,
                    min=512,
                    max=2_147_483_647,
                    optional=True,
                    tooltip="Text and media context size.",
                ),
                IO.Int.Input(
                    "n_batch",
                    default=512,
                    min=1,
                    max=2_147_483_647,
                    optional=True,
                    tooltip="Prompt processing batch size.",
                ),
                IO.Int.Input(
                    "gpu_layers",
                    default=-1,
                    min=-1,
                    max=2_147_483_647,
                    optional=True,
                    tooltip="-1 loads all to GPU",
                ),
                IO.Boolean.Input("verbose", default=False, optional=True),
            ],
            outputs=[
                IO.String.Output(
                    id="response_text",
                    display_name="Response Text",
                ),
                IO.String.Output(
                    id="tool_calls_json",
                    display_name="Tool Calls JSON",
                ),
                MESSAGES.Output(
                    id="messages",
                    display_name="Messages",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: str,
        mmproj: str,
        messages: Messages,
        tools_json: str = "[]",
        tool_choice: str = "auto",
        thinking: bool = True,
        chat_template_kwargs: str = "{}",
        temperature: float = 1.0,
        max_output_tokens: int = 1024,
        n_ctx: int = 8192,
        n_batch: int = 512,
        gpu_layers: int = -1,
        verbose: bool = False,
    ) -> IO.NodeOutput:
        tools = _tools_to_openai_chat(_parse_tools(tools_json))
        template_kwargs = (
            json.loads(chat_template_kwargs) if chat_template_kwargs else {}
        )
        if not isinstance(template_kwargs, dict):
            raise TypeError("`chat_template_kwargs` must decode to a JSON object.")
        if mmproj == "None" and any(message.files for message in messages):
            raise ValueError("Media messages require an MTMD projector.")
        response = _assistant_message(
            _run_completion(
                {
                    "model_path": folder_paths.get_full_path_or_raise(_MODEL_FOLDER, model),
                    "mmproj_path": (
                        None
                        if mmproj == "None"
                        else folder_paths.get_full_path_or_raise(_MODEL_FOLDER, mmproj)
                    ),
                    "messages": _messages_to_openai_chat(messages),
                    "tools": tools,
                    "tool_choice": tool_choice if tools else None,
                    "thinking": thinking,
                    "chat_template_kwargs": template_kwargs,
                    "temperature": temperature,
                    "max_tokens": max_output_tokens or None,
                    "n_ctx": n_ctx,
                    "n_batch": n_batch,
                    "gpu_layers": gpu_layers,
                    "verbose": verbose,
                }
            )
        )
        return IO.NodeOutput(
            response.text,
            json.dumps(
                [call.model_dump(mode="json", exclude_none=True) for call in response.tool_calls],
                indent=2,
            ),
            [*messages, response],
        )
