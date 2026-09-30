import asyncio
import json
import os
from typing import Any, ClassVar

import ai
import comfy.model_management
from ai.providers.openai import OpenAIChatCompletionsProtocol, OpenAIResponsesProtocol
from ai.providers.openai import protocol as openai_protocol
from ai.types import media as ai_media
from comfy_api.latest import IO
from dotenv import find_dotenv, load_dotenv

from .message_node import MESSAGES, Messages
from .secret_store import resolve_secret

load_dotenv(find_dotenv())

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]

# ai 0.5.0 recognizes video FileParts but omits the Responses input_video mapping.
_file_part_to_responses = openai_protocol._file_part_to_responses
async def _file_part_to_responses_with_video(
    part: ai.messages.FilePart,
) -> dict[str, Any]:
    if part.media_type.startswith("video/"):
        return {
            "type": "input_video",
            "video_url": ai_media.data_to_data_url(part.data, part.media_type),
        }
    return await _file_part_to_responses(part)
openai_protocol._file_part_to_responses = _file_part_to_responses_with_video


def _parse_tools(tools_json: str) -> list[ai.tools.Tool]:
    return [_parse_tool(tool) for tool in json.loads(tools_json)] if tools_json else []


def _parse_tool(tool: dict[str, JsonValue]) -> ai.tools.Tool:
    if "kind" in tool:
        return ai.tools.Tool.model_validate(tool)
    if tool.get("type") == "function":
        spec = tool.get("function", tool)
        return ai.tools.Tool(
            kind="function",
            name=spec.get("name", ""),
            spec=ai.tools.ToolSpec(
                description=spec.get("description", ""),
                params=spec.get("parameters", {}),
            ),
        )
    tool_type = tool.get("type", "")
    return ai.tools.Tool(
        kind="provider",
        name=tool_type,
        tool_config=ai.tools.ToolConfig(
            id=f"openai.{tool_type}",
            args={key: value for key, value in tool.items() if key != "type"},
        ),
    )


class _OpenAINode(IO.ComfyNode):
    node_id: ClassVar[str]
    display_name: ClassVar[str]
    protocol: ClassVar[type[OpenAIResponsesProtocol] | type[OpenAIChatCompletionsProtocol]]

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            inputs=[
                IO.String.Input("api_base", default="https://api.openai.com/v1"),
                IO.String.Input("api_key", default=""),
                IO.String.Input("model", default=""),
                MESSAGES.Input("messages", optional=False),
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
                ),
                IO.Float.Input(
                    "temperature", default=1.0, min=0.0, max=2.0, step=0.01, optional=True
                ),
                IO.Int.Input(
                    "n_ctx",
                    default=0,
                    min=0,
                    tooltip="Server-specific context size. 0 omits it.",
                    max=2_147_483_647,
                    optional=True,
                ),
                IO.Int.Input(
                    "max_output_tokens",
                    default=16354,
                    min=0,
                    max=2_147_483_647,
                    optional=True,
                ),
                IO.String.Input("extra_body", default="{}", multiline=True, optional=True),
            ],
            outputs=[
                IO.String.Output(id="response_text", display_name="Response Text"),
                IO.String.Output(id="tool_calls_json", display_name="Tool Calls JSON"),
                MESSAGES.Output(id="messages", display_name="Messages"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        api_base: str,
        api_key: str | None,
        model: str,
        messages: Messages,
        tools_json: str = "[]",
        temperature: float = 1.0,
        max_output_tokens: int = 16354,
        n_ctx: int = 0,
        tool_choice: str = "auto",
        extra_body: str = "{}",
    ) -> IO.NodeOutput:
        tools = _parse_tools(tools_json)
        provider = ai.get_provider(
            "openai",
            base_url=api_base,
            api_key=resolve_secret(api_key) if api_key else (
                (os.environ.get("OPENROUTER_API_KEY") if "openrouter.ai" in api_base else None)
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("OAI_API_KEY")
                or None
            ),
            protocol=cls.protocol(),
        )
        try:
            async with ai.stream(
                ai.Model(id=model, provider=provider),
                messages,
                tools=tools or None,
                params=ai.InferenceRequestParams(
                    sampling={
                        ai.TemperatureSamplerParams: ai.TemperatureSamplerParams(
                            temperature=temperature
                        )
                    },

                    output=(
                        ai.OutputParams(max_tokens=max_output_tokens)
                        if max_output_tokens > 0
                        else None
                    ),
                    tool_calling=(
                        ai.ToolCallingParams(tool_choice=ai.ToolChoiceMode(tool_choice))
                        if tools and tool_choice != "auto"
                        else None
                    ),
                    extra_body={
                        **(json.loads(extra_body) if extra_body else {}),
                        **({"n_ctx": n_ctx} if n_ctx else {}),
                    } or None,
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
                message = stream.message
        finally:
            await provider.aclose()
        return IO.NodeOutput(
            message.text,
            json.dumps(
                [call.model_dump(mode="json", exclude_none=True) for call in message.tool_calls],
                indent=2,
            ),
            [*messages, message],
        )


class OpenAIResponse(_OpenAINode):
    node_id = "OpenAIResponse"
    display_name = "OpenAI Response"
    protocol = OpenAIResponsesProtocol


class OpenAIChatCompletion(_OpenAINode):
    node_id = "OpenAICompletion"
    display_name = "OpenAI Chat Completion"
    protocol = OpenAIChatCompletionsProtocol
