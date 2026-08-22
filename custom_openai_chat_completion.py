import json
import os
from urllib.parse import urlsplit, urlunsplit

import ai
import comfy.model_management
from ai.providers.openai import (
    OpenAIChatCompletionsProtocol,
    OpenAIResponsesProtocol,
)
from comfy_api.latest import IO
from comfy_api_nodes.util.conversions import (
    audio_to_base64_string,
    tensor_to_data_uri,
    video_to_base64_string,
)
from dotenv import load_dotenv
from torch import Tensor

from .secret_store import resolve_secret

load_dotenv()

type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)


def _parse_request(
    api_key: str,
    model: str,
    messages_json: str | None,
    tools_json: str,
) -> tuple[str, list[ai.messages.Message], list[ai.tools.Tool]]:
    api_key = (
        resolve_secret(api_key) if api_key else os.environ.get("OPENAI_API_KEY", "")
    )
    if not api_key:
        raise ValueError("`api_key` is required.")

    if not model:
        raise ValueError("`model` is required.")

    return api_key, _parse_messages(messages_json), _parse_tools(tools_json)


def _parse_messages(messages_json: str | None) -> list[ai.messages.Message]:
    if not messages_json:
        raise ValueError("`messages_json` is required.")

    items = json.loads(messages_json)
    if not isinstance(items, list) or not items:
        raise ValueError("`messages_json` must decode to a non-empty list.")
    return [ai.messages.Message.model_validate(item) for item in items]


def _parse_tools(tools_json: str) -> list[ai.tools.Tool]:
    return [_parse_tool(tool) for tool in json.loads(tools_json)] if tools_json else []


def _parse_tool(tool: dict[str, JsonValue]) -> ai.tools.Tool:
    """Accept AI SDK tool JSON or OpenAI function/tool definitions."""
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


def _dump_messages(messages: list[ai.messages.Message]) -> str:
    return json.dumps(
        [m.model_dump(mode="json", exclude_none=True) for m in messages],
        indent=2,
    )


def _completion_output(
    messages: list[ai.messages.Message], message: ai.messages.Message
) -> IO.NodeOutput:
    return IO.NodeOutput(
        message.text,
        json.dumps(
            [
                tool_call.model_dump(mode="json", exclude_none=True)
                for tool_call in message.tool_calls
            ],
            indent=2,
        ),
        _dump_messages(messages + [message]),
    )


def _tool_result(
    messages: list[ai.messages.Message], content: str
) -> ai.messages.Message:
    call = next(
        (
            part
            for message in reversed(messages)
            for part in reversed(message.tool_calls)
        ),
        None,
    )
    if not call:
        raise ValueError("A tool message requires a preceding tool call.")
    return ai.tool_message(
        tool_call_id=call.tool_call_id,
        result=content,
        tool_name=call.tool_name,
    )


def _media_part(
    attachment: IO.Image.Type | IO.Video.Type | IO.Audio.Type,
) -> ai.messages.FilePart:
    if isinstance(attachment, Tensor):
        return ai.file_part(
            tensor_to_data_uri(attachment, mime_type="image/png"),
            media_type="image/png",
        )
    if isinstance(attachment, dict):
        return ai.file_part(
            f"data:audio/mp4;base64,{audio_to_base64_string(attachment)}",
            media_type="audio/mp4",
        )
    return ai.file_part(
        f"data:video/mp4;base64,{video_to_base64_string(attachment)}",
        media_type="video/mp4",
    )


def _normalize_api_base(api_base: str) -> str:
    parsed = urlsplit(api_base)
    return urlunsplit(
        parsed._replace(
            path=parsed.path.rstrip("/")
            .removesuffix("/responses")
            .removesuffix("/chat/completions")
        )
    )


def _inference_params(
    temperature: float,
    max_output_tokens: int,
    tool_choice: str,
    extra_body: str,
    tools: list[ai.tools.Tool],
) -> ai.InferenceRequestParams:
    return ai.InferenceRequestParams(
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
        extra_body=json.loads(extra_body) if extra_body else None,
    )


async def _run_completion(
    api_base: str,
    api_key: str,
    model_id: str,
    messages: list[ai.messages.Message],
    tools: list[ai.tools.Tool],
    temperature: float,
    max_output_tokens: int,
    tool_choice: str,
    extra_body: str,
    protocol: OpenAIResponsesProtocol | OpenAIChatCompletionsProtocol,
) -> ai.messages.Message:
    provider = ai.get_provider(
        "openai",
        base_url=_normalize_api_base(api_base),
        api_key=api_key,
        protocol=protocol,
    )
    try:
        async with ai.stream(
            ai.Model(id=model_id, provider=provider),
            messages,
            tools=tools or None,
            params=_inference_params(
                temperature, max_output_tokens, tool_choice, extra_body, tools
            ),
        ) as stream:
            async for _ in stream:
                comfy.model_management.throw_exception_if_processing_interrupted()
            return stream.message
    finally:
        await provider.aclose()


class CustomOpenAIResponse(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="CustomOpenAIResponse",
            display_name="OpenAI Response",
            category="api node/text/OpenAI",
            description="Stream an assistant turn via the AI SDK (OpenAI Responses protocol).",
            inputs=[
                IO.String.Input(
                    "api_base",
                    default="https://api.openai.com/v1",
                    tooltip="Base URL for the OpenAI-compatible API.",
                ),
                IO.String.Input(
                    "api_key",
                    default="",
                    tooltip="API key for Authorization.",
                ),
                IO.String.Input(
                    "model",
                    default="",
                    tooltip="Model name.",
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
                    tooltip="Tools array JSON: AI SDK tools or OpenAI function/tool definitions.",
                ),
                IO.String.Input(
                    "extra_body",
                    default="{}",
                    multiline=True,
                    optional=True,
                    tooltip="Provider-specific request body fields as a JSON object.",
                ),
                IO.Combo.Input(
                    "tool_choice",
                    options=["auto", "none", "required"],
                    default="auto",
                    optional=True,
                    tooltip="Tool choice policy. Applies when tools are provided.",
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
                    optional=True,
                    tooltip="Max output tokens. 0 to omit.",
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
    async def execute(
        cls,
        api_base: str,
        api_key: str,
        model: str,
        messages_json: str | None,
        tools_json: str = "[]",
        temperature: float = 1.0,
        max_output_tokens: int = 1024,
        tool_choice: str = "auto",
        extra_body: str = "{}",
    ) -> IO.NodeOutput:
        api_key, messages, tools = _parse_request(
            api_key, model, messages_json, tools_json
        )

        message = await _run_completion(
            api_base,
            api_key,
            model,
            messages,
            tools,
            temperature,
            max_output_tokens,
            tool_choice,
            extra_body,
            OpenAIResponsesProtocol(),
        )

        return _completion_output(messages, message)


class MessageAppend(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="MessageAppend",
            display_name="Message Append",
            category="api node/text/OpenAI",
            description="Append one message to an AI SDK message array.",
            inputs=[
                IO.String.Input(
                    "messages_json",
                    default="[]",
                    multiline=True,
                    socketless=False,
                    optional=True,
                    force_input=True,
                    tooltip="Existing AI SDK messages JSON array.",
                ),
                IO.Combo.Input(
                    "role",
                    options=["system", "user", "assistant", "tool"],
                    default="user",
                    tooltip="Role for the appended message.",
                ),
                IO.String.Input(
                    "content",
                    default="",
                    multiline=True,
                    socketless=True,
                    force_input=False,
                    tooltip="Text content or tool result.",
                ),
                IO.Autogrow.Input(
                    "media",
                    template=IO.Autogrow.TemplatePrefix(
                        input=IO.MultiType.Input(
                            "media",
                            types=[IO.Image, IO.Video, IO.Audio],
                            tooltip="Image, video, or audio to include in the message.",
                        ),
                        prefix="media",
                        min=0,
                    ),
                    tooltip="Optional image, video, and audio attachments.",
                ),
            ],
            outputs=[
                IO.String.Output(
                    id="messages_json",
                    display_name="Messages JSON",
                    tooltip="JSON array of AI SDK messages.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        messages_json: str = "[]",
        role: str = "user",
        content: str = "",
        media: IO.Autogrow.Type | None = None,
    ) -> IO.NodeOutput:
        messages = (
            [
                ai.messages.Message.model_validate(item)
                for item in json.loads(messages_json)
            ]
            if messages_json
            else []
        )

        if role == "tool":
            messages.append(_tool_result(messages, content))
            return IO.NodeOutput(_dump_messages(messages))

        parts: list[str | ai.messages.FilePart] = []
        if content:
            parts.append(content)
        for attachment in (media or {}).values():
            parts.append(_media_part(attachment))
        messages.append(ai.message(*parts, role=role))
        return IO.NodeOutput(_dump_messages(messages))


class CustomOpenAICompletion(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="CustomOpenAICompletion",
            display_name="OpenAI Chat Completion",
            category="api node/text/OpenAI",
            description="Stream an assistant turn via the AI SDK (OpenAI Chat Completions protocol).",
            inputs=[
                IO.String.Input(
                    "api_base",
                    default="https://api.openai.com/v1",
                    tooltip="Base URL for the OpenAI-compatible API.",
                ),
                IO.String.Input(
                    "api_key",
                    default="",
                    tooltip="API key for Authorization.",
                ),
                IO.String.Input(
                    "model",
                    default="",
                    tooltip="Model name.",
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
                    tooltip="Tools array JSON: AI SDK tools or OpenAI function/tool definitions.",
                ),
                IO.String.Input(
                    "extra_body",
                    default="{}",
                    multiline=True,
                    optional=True,
                    tooltip="Provider-specific request body fields as a JSON object.",
                ),
                IO.Combo.Input(
                    "tool_choice",
                    options=["auto", "none", "required"],
                    default="auto",
                    optional=True,
                    tooltip="Tool choice policy. Applies when tools are provided.",
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
                    optional=True,
                    tooltip="Max output tokens. 0 to omit.",
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
    async def execute(
        cls,
        api_base: str,
        api_key: str,
        model: str,
        messages_json: str | None,
        tools_json: str = "[]",
        temperature: float = 1.0,
        max_output_tokens: int = 1024,
        tool_choice: str = "auto",
        extra_body: str = "{}",
    ) -> IO.NodeOutput:
        api_key, messages, tools = _parse_request(
            api_key, model, messages_json, tools_json
        )

        message = await _run_completion(
            api_base,
            api_key,
            model,
            messages,
            tools,
            temperature,
            max_output_tokens,
            tool_choice,
            extra_body,
            OpenAIChatCompletionsProtocol(),
        )

        return _completion_output(messages, message)
