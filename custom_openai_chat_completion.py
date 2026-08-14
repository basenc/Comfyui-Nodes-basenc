import json
import os
from typing import Any

import ai
from ai.providers.openai import (
    OpenAIChatCompletionsProtocol,
    OpenAIResponsesProtocol,
)
from dotenv import load_dotenv

import comfy.model_management
from comfy_api.latest import IO

load_dotenv()


def _parse_request(
    api_key: str,
    model: str,
    messages_json: str | None,
    tools_json: str,
) -> tuple[str, list[ai.messages.Message], list[ai.tools.Tool]]:
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("`api_key` is required.")

    if not model:
        raise ValueError("`model` is required.")

    if not messages_json:
        raise ValueError("`messages_json` is required.")

    items = json.loads(messages_json)
    if not isinstance(items, list) or not items:
        raise ValueError("`messages_json` must decode to a non-empty list.")

    messages = [ai.messages.Message.model_validate(item) for item in items]
    tools = (
        [_parse_tool(tool) for tool in json.loads(tools_json)]
        if tools_json
        else []
    )
    return api_key, messages, tools


def _parse_tool(tool: dict[str, Any]) -> ai.tools.Tool:
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


async def _run_completion(
    api_base: str,
    api_key: str,
    model_id: str,
    messages: list[ai.messages.Message],
    tools: list[ai.tools.Tool],
    temperature: float,
    max_output_tokens: int,
    tool_choice: str,
    protocol: OpenAIResponsesProtocol | OpenAIChatCompletionsProtocol,
) -> ai.messages.Message:
    provider = ai.get_provider(
        "openai",
        base_url=api_base.rstrip("/")
        .removesuffix("/responses")
        .removesuffix("/chat/completions"),
        api_key=api_key,
        protocol=protocol,
    )
    params = ai.InferenceRequestParams(
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
    )
    try:
        async with ai.stream(
            ai.Model(id=model_id, provider=provider),
            messages,
            tools=tools or None,
            params=params,
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
            OpenAIResponsesProtocol(),
        )

        return IO.NodeOutput(
            message.text,
            json.dumps(
                [
                    tc.model_dump(mode="json", exclude_none=True)
                    for tc in message.tool_calls
                ],
                indent=2,
            ),
            _dump_messages(messages + [message]),
        )


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
                IO.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Optional image tensor.",
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
        image=None,
    ) -> IO.NodeOutput:
        from comfy_api_nodes.util.conversions import tensor_to_data_uri

        messages = (
            [ai.messages.Message.model_validate(item) for item in json.loads(messages_json)]
            if messages_json
            else []
        )

        if role == "tool":
            call = next(
                (
                    part
                    for message in reversed(messages)
                    for part in reversed(message.tool_calls)
                ),
                None,
            )
            if not call:
                raise ValueError(
                    "A tool message requires a preceding tool call."
                )
            messages.append(
                ai.tool_message(
                    tool_call_id=call.tool_call_id,
                    result=content,
                    tool_name=call.tool_name,
                )
            )
            return IO.NodeOutput(_dump_messages(messages))

        parts: list[str | ai.messages.FilePart] = []
        if content:
            parts.append(content)
        if image is not None:
            parts.append(
                ai.file_part(
                    tensor_to_data_uri(image, mime_type="image/png"),
                    media_type="image/png",
                )
            )
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
            OpenAIChatCompletionsProtocol(),
        )

        return IO.NodeOutput(
            message.text,
            json.dumps(
                [
                    tc.model_dump(mode="json", exclude_none=True)
                    for tc in message.tool_calls
                ],
                indent=2,
            ),
            _dump_messages(messages + [message]),
        )
