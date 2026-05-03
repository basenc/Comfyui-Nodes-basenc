import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

import comfy.model_management
from comfy_api.latest import IO

load_dotenv()


class CustomOpenAIResponse(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="CustomOpenAIResponse",
            display_name="OpenAI Response",
            category="api node/text/OpenAI",
            description="Send input items to OpenAI Responses API.",
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
                    "input_json",
                    multiline=True,
                    optional=False,
                    socketless=False,
                    force_input=True,
                    tooltip="JSON array of Responses API input items.",
                ),
                IO.String.Input(
                    "tools_json",
                    default="[]",
                    multiline=True,
                    optional=True,
                    socketless=False,
                    force_input=False,
                    tooltip="Tools array JSON.",
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
                IO.Float.Input(
                    "timeout_seconds",
                    default=60.0,
                    min=1.0,
                    step=1.0,
                    optional=True,
                    tooltip="Request timeout in seconds.",
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
                    tooltip="function_call items from output.",
                ),
                IO.String.Output(
                    id="output_json",
                    display_name="Output JSON",
                    tooltip="Input + Response conversation JSON.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        api_base: str,
        api_key: str,
        model: str,
        input_json: str | None,
        tools_json: str = "[]",
        temperature: float = 1.0,
        max_output_tokens: int = 1024,
        timeout_seconds: float = 60.0,
    ) -> IO.NodeOutput:
        api_key = api_key or os.environ["OPENAI_API_KEY"]
        if not api_key:
            raise ValueError("`api_key` is required.")

        if not model:
            raise ValueError("`model` is required.")

        if not input_json:
            raise ValueError("`input_json` is required.")

        input_items = json.loads(input_json)
        if not isinstance(input_items, list) or not input_items:
            raise ValueError("`input_json` must decode to a non-empty list.")

        tools = json.loads(tools_json) if tools_json else []

        kwargs: Dict[str, Any] = {
            "model": model,
            "input": input_items,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
        if max_output_tokens > 0:
            kwargs["max_output_tokens"] = max_output_tokens

        client = OpenAI(
            api_key=api_key,
            base_url=api_base.rstrip("/").removesuffix("/responses"),
        )

        data = None
        for event in client.responses.create(stream=True, timeout=timeout_seconds, **kwargs):
            comfy.model_management.throw_exception_if_processing_interrupted()
            if event.type == "response.completed":
                data = event.response.model_dump()

        return IO.NodeOutput(
            "".join(
                c["text"]
                for item in data["output"] if item["type"] == "message"
                for c in item["content"] if c["type"] == "output_text"
            ),
            json.dumps(
                [item for item in data["output"] if item["type"] == "function_call"],
                indent=2,
            ),
            json.dumps(list(input_items) + data["output"], indent=2),
        )


class ResponseInputAppend(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="ResponseInputAppend",
            display_name="Response Input Append",
            category="api node/text/OpenAI",
            description="Append one input item to a Responses API input array.",
            inputs=[
                IO.String.Input(
                    "input_json",
                    default="[]",
                    multiline=True,
                    socketless=False,
                    optional=True,
                    force_input=True,
                    tooltip="Existing input JSON array.",
                ),
                IO.Combo.Input(
                    "role",
                    options=["system", "user", "developer", "function_call_output"],
                    default="user",
                    tooltip="Role or type for this input item.",
                ),
                IO.String.Input(
                    "content",
                    default="",
                    multiline=True,
                    socketless=True,
                    force_input=False,
                    tooltip="Text content or function call output.",
                ),
                IO.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Optional image tensor.",
                ),
            ],
            outputs=[
                IO.String.Output(
                    id="input_json",
                    display_name="Input JSON",
                    tooltip="JSON array of input items.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        input_json: str = "[]",
        role: str = "user",
        content: str = "",
        image=None,
    ) -> IO.NodeOutput:
        from comfy_api_nodes.util.conversions import tensor_to_data_uri

        items = list(json.loads(input_json)) if input_json else []

        if role == "function_call_output":
            call_id = next(
                (
                    item["call_id"]
                    for item in reversed(items)
                    if "type" in item and item["type"] == "function_call"
                ),
                None,
            )
            if not call_id:
                raise ValueError(
                    "function_call_output requires a preceding function_call item."
                )
            items.append(
                {"type": "function_call_output", "call_id": call_id, "output": content}
            )
            return IO.NodeOutput(json.dumps(items, indent=2))

        content_parts: List[Dict[str, Any]] = []
        if content:
            content_parts.append({"type": "input_text", "text": content})

        if image is not None:
            content_parts.append(
                {
                    "type": "input_image",
                    "image_url": tensor_to_data_uri(image, mime_type="image/png"),
                    "detail": "auto",
                }
            )

        items.append({"role": role, "content": content_parts or content})
        return IO.NodeOutput(json.dumps(items, indent=2))
