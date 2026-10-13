import asyncio
import json
from pathlib import Path

import ai
import comfy.model_management
import folder_paths
from ai.providers.openai import OpenAIChatCompletionsProtocol
from comfy_api.latest import IO

from .llama_cpp_backend import _MODEL_FOLDER, _llama_server, _model_options
from .message_node import MESSAGES, Messages
from .openai_chat_node import _parse_tools



async def _run_llama_server(
    *,
    model_path: str,
    mmproj_path: str | None,
    n_ctx: str,
    gpu_layers: str,
    llama_server_args: str,
    mtp_model: str,
    messages: Messages,
    tools: list[ai.tools.Tool],
    tool_choice: str,
    temperature: float,
    max_tokens: int | None,
    extra_body: dict[str, object] | None,
) -> ai.messages.Message:
    async with _llama_server(
        model_path=model_path,
        mmproj_path=mmproj_path,
        n_ctx=n_ctx,
        gpu_layers=gpu_layers,
        llama_server_args=llama_server_args,
        mtp_model=mtp_model,
    ) as api_base:
        provider = ai.get_provider(
            "openai",
            base_url=f"{api_base}/v1",
            api_key="sk-noop",
            protocol=OpenAIChatCompletionsProtocol(),
        )
        try:
            async with ai.stream(
                ai.Model(id=Path(model_path).name, provider=provider),
                messages,
                tools=tools or None,
                params=ai.InferenceRequestParams(
                    sampling={
                        ai.TemperatureSamplerParams: ai.TemperatureSamplerParams(
                            temperature=temperature
                        )
                    },
                    output=(
                        ai.OutputParams(max_tokens=max_tokens)
                        if max_tokens is not None
                        else None
                    ),
                    tool_calling=(
                        ai.ToolCallingParams(
                            tool_choice=ai.ToolChoiceMode(tool_choice)
                        )
                        if tools and tool_choice != "auto"
                        else None
                    ),
                    extra_body=extra_body,
                ),
            ) as stream:
                events = aiter(stream)
                while True:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    next_event = asyncio.create_task(anext(events))
                    try:
                        while not next_event.done():
                            await asyncio.wait({next_event}, timeout=0.25)
                            comfy.model_management.throw_exception_if_processing_interrupted()
                        await next_event
                    except StopAsyncIteration:
                        break
                    finally:
                        if not next_event.done():
                            next_event.cancel()
                        await asyncio.gather(next_event, return_exceptions=True)
                return stream.message
        finally:
            await provider.aclose()


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
                IO.Combo.Input(
                    "mtp_model",
                    options=["None", *_model_options(mmproj=False)],
                    default="None",
                    optional=True,
                    tooltip="MTP draft GGUF. None disables MTP speculative decoding.",
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
                IO.String.Input(
                    "n_ctx",
                    default="auto",
                    optional=True,
                    tooltip="Context size: auto fits available memory, or enter a token count.",
                ),

                IO.String.Input(
                    "gpu_layers",
                    default="auto",
                    optional=True,
                    tooltip="GPU layers: auto, all, or a number. Memory fitting is enabled.",
                ),
                IO.String.Input(
                    "llama_server_args",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Extra llama-server flags. Quoting is supported; node flags take priority.",
                ),
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
    async def execute(
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
        n_ctx: str = "auto",
        gpu_layers: str = "auto",
        llama_server_args: str = "",
        mtp_model: str = "None",
    ) -> IO.NodeOutput:
        tools = _parse_tools(tools_json)
        template_kwargs = (
            json.loads(chat_template_kwargs) if chat_template_kwargs else {}
        )
        if not isinstance(template_kwargs, dict):
            raise TypeError("`chat_template_kwargs` must decode to a JSON object.")
        if thinking:
            template_kwargs.setdefault("enable_thinking", True)
        if mmproj == "None" and any(message.files for message in messages):
            raise ValueError("Media messages require an MTMD projector.")
        model_path = folder_paths.get_full_path_or_raise(_MODEL_FOLDER, model)
        response = await _run_llama_server(
            model_path=model_path,
            mmproj_path=(
                None
                if mmproj == "None"
                else folder_paths.get_full_path_or_raise(_MODEL_FOLDER, mmproj)
            ),
            n_ctx=n_ctx,
            gpu_layers=gpu_layers,
            llama_server_args=llama_server_args,
            mtp_model=mtp_model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_output_tokens or None,
            extra_body={"chat_template_kwargs": template_kwargs}
            if template_kwargs
            else None,
        )
        return IO.NodeOutput(
            response.text,
            json.dumps(
                [call.model_dump(mode="json", exclude_none=True) for call in response.tool_calls],
                indent=2,
            ),
            [*messages, response],
        )