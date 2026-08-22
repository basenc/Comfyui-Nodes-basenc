import gc
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import ai
import comfy.model_management
import folder_paths
from ai.types import media
from ai.types.usage import Usage
from comfy_api.latest import IO

from .custom_openai_chat_completion import (
    JsonValue,
    _completion_output,
    _parse_messages,
    _parse_tools,
)

type LoraStack = list[tuple[str, float]]
type ChatMessage = dict[str, JsonValue]
type ChatTool = dict[str, JsonValue]

_MODEL_FOLDER = "basenc_llama_cpp_models"
_LORA_FOLDER = "basenc_llama_cpp_loras"
_LORA_STACK_IO = IO.Custom("BASENC_LLAMA_CPP_LORAS")
_MODELS_DIR = Path(folder_paths.models_dir)
_NO_MMPROJ = "None"
_MAX_INT = 2_147_483_647


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
_register_model_folders(
    _LORA_FOLDER,
    {".gguf", ".safetensors"},
    _MODELS_DIR / "LLM",
    _MODELS_DIR / "text_encoders",
    _MODELS_DIR / "loras",
)


def _model_options(mmproj: bool) -> list[str]:
    return [
        name
        for name in folder_paths.get_filename_list(_MODEL_FOLDER)
        if ("mmproj" in Path(name).name.casefold()) == mmproj
    ]


def _resolve_loras(loras: LoraStack | None) -> list[tuple[str, float]]:
    selections = [
        (folder_paths.get_full_path_or_raise(_LORA_FOLDER, name), strength)
        for name, strength in loras or []
    ]
    if len({path for path, _ in selections}) != len(selections):
        raise ValueError("A LoRA can only appear once in a stack.")
    if not all(math.isfinite(strength) for _, strength in selections):
        raise ValueError("LoRA strengths must be finite.")
    return selections


def _run_worker(request: dict[str, JsonValue]) -> dict[str, JsonValue]:
    comfy.model_management.unload_all_models()
    gc.collect()
    comfy.model_management.soft_empty_cache(force=True)
    with tempfile.TemporaryDirectory(prefix="llama-cpp-") as directory:
        request_path = Path(directory) / "request.json"
        response_path = Path(directory) / "response.json"
        stderr_path = Path(directory) / "stderr.log"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        with stderr_path.open("w", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("llama_cpp_worker.py")),
                    str(request_path),
                    str(response_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
            )
            while process.poll() is None:
                try:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                except comfy.model_management.InterruptProcessingException:
                    process.terminate()
                    process.wait()
                    raise
                time.sleep(0.1)
        if process.returncode != 0:
            raise RuntimeError(
                f"llama.cpp worker crashed with exit code {process.returncode}.\n"
                f"{stderr_path.read_text(encoding='utf-8')}"
            )
        return json.loads(response_path.read_text(encoding="utf-8"))


def _tool_result_content(value: JsonValue | ai.messages.ContentOutput) -> str:
    if isinstance(value, ai.messages.ContentOutput):
        raise TypeError("MTMD does not support multipart tool results.")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), default=str)


def _messages_to_llama(
    messages: list[ai.messages.Message],
) -> list[ChatMessage]:
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


def _tools_to_llama(tools: list[ai.tools.Tool]) -> list[ChatTool]:
    result: list[ChatTool] = []
    for tool in tools:
        if tool.kind != "function" or tool.spec is None:
            raise ValueError("llama.cpp supports function tools only.")
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


class LlamaCppLora(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="LlamaCppLora",
            display_name="llama.cpp LoRA",
            category="api node/text/llama.cpp",
            inputs=[
                _LORA_STACK_IO.Input(
                    "loras",
                    optional=True,
                    tooltip="Optional preceding llama.cpp LoRA stack.",
                ),
                IO.Combo.Input(
                    "lora",
                    options=folder_paths.get_filename_list(_LORA_FOLDER),
                    tooltip="LoRA from models/LLM, models/text_encoders, or models/loras.",
                ),
                IO.Float.Input(
                    "strength",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    tooltip="LoRA adapter scale.",
                ),
            ],
            outputs=[
                _LORA_STACK_IO.Output(
                    id="loras",
                    display_name="LoRAs",
                )
            ],
        )

    @classmethod
    def execute(
        cls,
        lora: str,
        strength: float,
        loras: LoraStack | None = None,
    ) -> IO.NodeOutput:
        return IO.NodeOutput([*(loras or []), (lora, strength)])


class LlamaCppCompletion(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="LlamaCppCompletion",
            display_name="llama.cpp Chat Completion",
            category="api node/text/llama.cpp",
            description="Run a local llama.cpp chat completion with optional MTMD.",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=_model_options(mmproj=False),
                    tooltip="Text-model GGUF from models/LLM or models/text_encoders.",
                ),
                IO.Combo.Input(
                    "mmproj",
                    options=[_NO_MMPROJ, *_model_options(mmproj=True)],
                    default=_NO_MMPROJ,
                    tooltip="Optional MTMD projector. None runs text-only chat.",
                ),
                _LORA_STACK_IO.Input(
                    "loras",
                    optional=True,
                    tooltip="Optional llama.cpp LoRA stack.",
                ),
                IO.String.Input(
                    "messages_json",
                    multiline=True,
                    optional=False,
                    socketless=False,
                    force_input=True,
                    tooltip="JSON array of AI SDK messages.",
                ),
                IO.String.Input(
                    "tools_json",
                    default="[]",
                    multiline=True,
                    optional=True,
                    socketless=False,
                    force_input=False,
                    tooltip="Tools array JSON: AI SDK tools or OpenAI function definitions.",
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
                    max=_MAX_INT,
                    optional=True,
                    tooltip="Max output tokens. 0 to use the remaining context.",
                ),
                IO.Int.Input(
                    "n_ctx",
                    default=8192,
                    min=512,
                    max=_MAX_INT,
                    optional=True,
                    tooltip="Text and media context size.",
                ),
                IO.Int.Input(
                    "n_batch",
                    default=512,
                    min=1,
                    max=_MAX_INT,
                    optional=True,
                    tooltip="Prompt processing batch size.",
                ),
                IO.Int.Input(
                    "gpu_layers",
                    default=-1,
                    min=-1,
                    max=_MAX_INT,
                    optional=True,
                    tooltip="GPU layers: -1 for all, 0 for CPU, positive for partial offload.",
                ),
            ],
            outputs=[
                IO.String.Output(
                    id="response_text",
                    display_name="Response Text",
                    tooltip="Aggregated output text.",
                ),
                IO.String.Output(
                    id="tool_calls_json",
                    display_name="Tool Calls JSON",
                    tooltip="JSON array of tool call parts.",
                ),
                IO.String.Output(
                    id="output_json",
                    display_name="Output JSON",
                    tooltip="Messages + assistant message JSON.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: str,
        mmproj: str,
        messages_json: str | None,
        loras: LoraStack | None = None,
        tools_json: str = "[]",
        tool_choice: str = "auto",
        thinking: bool = True,
        chat_template_kwargs: str = "{}",
        temperature: float = 1.0,
        max_output_tokens: int = 1024,
        n_ctx: int = 8192,
        n_batch: int = 512,
        gpu_layers: int = -1,
    ) -> IO.NodeOutput:
        messages = _parse_messages(messages_json)
        tools = _tools_to_llama(_parse_tools(tools_json))
        template_kwargs = (
            json.loads(chat_template_kwargs) if chat_template_kwargs else {}
        )
        if not isinstance(template_kwargs, dict):
            raise ValueError("`chat_template_kwargs` must decode to a JSON object.")
        if mmproj == _NO_MMPROJ and any(message.files for message in messages):
            raise ValueError("Media messages require an MTMD projector.")
        return _completion_output(
            messages,
            _assistant_message(
                _run_worker(
                    {
                        "model_path": folder_paths.get_full_path_or_raise(
                            _MODEL_FOLDER, model
                        ),
                        "mmproj_path": (
                            None
                            if mmproj == _NO_MMPROJ
                            else folder_paths.get_full_path_or_raise(
                                _MODEL_FOLDER, mmproj
                            )
                        ),
                        "loras": _resolve_loras(loras),
                        "messages": _messages_to_llama(messages),
                        "tools": tools,
                        "tool_choice": tool_choice if tools else None,
                        "thinking": thinking,
                        "chat_template_kwargs": template_kwargs,
                        "temperature": temperature,
                        "max_tokens": max_output_tokens or None,
                        "n_ctx": n_ctx,
                        "n_batch": n_batch,
                        "gpu_layers": gpu_layers,
                    }
                )
            ),
        )
