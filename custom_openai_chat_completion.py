import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from comfy_api.latest import IO

load_dotenv()


class CustomOpenAIChatCompletion(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="CustomOpenAIChatCompletion",
            display_name="Custom OpenAI Chat Completion",
            category="api node/text/OpenAI",
            description=(
                "Send a raw `messages` array to a chat completions endpoint with a custom "
                "API base URL and API key."
            ),
            inputs=[
                IO.String.Input(
                    "api_base",
                    default="https://api.openai.com/v1",
                    tooltip="Base URL for the OpenAI-compatible API (no trailing slash needed).",
                ),
                IO.String.Input(
                    "api_key",
                    default="",
                    tooltip="API key used for the Authorization header.",
                ),
                IO.String.Input(
                    "model",
                    default="gpt-4o-mini",
                    tooltip="Model name to send to the chat completion endpoint.",
                ),
                IO.String.Input(
                    "messages_json",
                    multiline=True,
                    optional=False,
                    socketless=False,
                    force_input=True,
                    tooltip="Raw JSON array of chat messages to send as the `messages` field.",
                ),
                IO.String.Input(
                    "tools_json",
                    default="[]",
                    multiline=True,
                    optional=True,
                    socketless=False,
                    force_input=False,
                    tooltip="Optional tools array JSON.",
                ),
                IO.Float.Input(
                    "temperature",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    optional=False,
                    tooltip="Optional temperature value to include in the request.",
                ),
                IO.Int.Input(
                    "max_tokens",
                    default=1024,
                    min=0,
                    optional=False,
                    tooltip="Optional max_tokens value. Set to 0 to omit.",
                ),
                IO.Float.Input(
                    "timeout_seconds",
                    default=60.0,
                    min=1.0,
                    step=1.0,
                    optional=False,
                    tooltip="Request timeout in seconds.",
                ),
            ],
            outputs=[
                IO.String.Output(
                    id="response_text",
                    display_name="Response Text",
                    tooltip="First choice message content from the API response.",
                ),
                IO.String.Output(
                    id="raw_json",
                    display_name="Response JSON",
                    tooltip="Full JSON response from the API.",
                ),
                IO.String.Output(
                    id="tool_calls_json",
                    display_name="Tool Calls JSON",
                    tooltip="tool_calls array from the first choice, if any.",
                ),
                IO.String.Output(
                    id="messages_json_out",
                    display_name="Messages JSON",
                    tooltip="Messages array you can feed to the next request (includes assistant message from this completion).",
                ),
            ],
        )

    @classmethod
    def _extract_text(cls, data: Dict[str, Any]) -> str:
        # --- Responses API ---
        if "output" in data or "output_text" in data:
            if "output_text" in data:
                return data["output_text"] or ""

            parts = []
            for item in data.get("output", []):
                t = item.get("type")

                # Skip non-text types
                if t in {
                    "function_call",
                    "tool_call",
                    "mcp_call",
                    "mcp_approval_request",
                    "mcp_list_tools",
                    "tool_output",
                    "reasoning",
                    "code_interpreter",
                    "computer_use",
                }:
                    continue

                # Message → content → text chunks
                if t == "message":
                    for c in item.get("content", []):
                        if c.get("type") in {"output_text", "input_text"}:
                            if c.get("text"):
                                parts.append(c["text"])

                # Standalone output_text type
                if t == "output_text" and item.get("text"):
                    parts.append(item["text"])

            return "".join(parts)

        # --- Chat Completions ---
        if data.get("object") == "chat.completion":
            choices = data.get("choices") or []
            if not choices:
                return ""
            return choices[0].get("message", {}).get("content") or ""

        # --- Legacy Completions ---
        choices = data.get("choices")
        if choices and isinstance(choices, list):
            return choices[0].get("text", "") or ""

        return ""

    @classmethod
    def _extract_tool_calls(cls, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        calls = []

        # --- Responses API ---
        if "output" in data:
            for item in data["output"]:
                t = item.get("type")
                if t in {"function_call", "tool_call", "mcp_call"}:
                    calls.append(
                        {
                            "type": t,
                            "id": item.get("id") or item.get("call_id"),
                            "function": {
                                "name": item.get("function", {}).get("name", ""),
                                "arguments": item.get("function", {}).get(
                                    "arguments", ""
                                ),
                            },
                        }
                    )
            return calls

        # --- Chat Completions ---
        if data.get("object") == "chat.completion":
            choices = data.get("choices") or []
            if not choices:
                return []

            msg = choices[0].get("message", {})
            tcs = msg.get("tool_calls") or []

            for tc in tcs:
                fn = tc.get("function", {})
                calls.append(
                    {
                        "type": tc.get("type", "function"),
                        "id": tc.get("id"),
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", ""),
                        },
                    }
                )
            return calls

        # --- Legacy Completions (none) ---
        return []

    @classmethod
    def execute(
        cls,
        api_base: str,
        api_key: str,
        model: str,
        messages_json: str | None,
        tools_json: str = "[]",
        temperature: float = 1.0,
        max_tokens: int = 256,
        timeout_seconds: float = 60.0,
    ) -> IO.NodeOutput:
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "API key is required; provide it via input or OPENAI_API_KEY."
            )

        if not messages_json:
            raise ValueError(
                "`messages_json` is required; provide a JSON-encoded non-empty list of messages."
            )
        messages = json.loads(messages_json)
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError(
                "`messages_json` must decode to a non-empty list of messages."
            )

        tools = json.loads(tools_json) if tools_json else []

        base = api_base.rstrip("/").removesuffix("/chat/completions")

        client = OpenAI(api_key=api_key, base_url=base)

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout_seconds,
            tools=tools if tools else None,
            max_tokens=max_tokens if max_tokens > 0 else None,
        )

        data: Dict[str, Any] = completion.model_dump()

        response_text = cls._extract_text(data)
        tool_calls = cls._extract_tool_calls(data)
        tool_calls_json = json.dumps(tool_calls, indent=2)
        raw_json = json.dumps(data, indent=2)

        messages_out = list(messages)
        messages_out.append(
            {
                "role": "assistant",
                **(
                    {"content": [{"type": "text", "text": response_text}]}
                    if response_text
                    else {}
                ),
                **({"tool_calls": tool_calls} if tool_calls else {}),
            }
        )

        messages_json_out = json.dumps(messages_out, indent=2)

        return IO.NodeOutput(
            response_text, raw_json, tool_calls_json, messages_json_out
        )


# --- Message builder nodes -------------------------------------------------


class ChatMessagesCreate(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="ChatMessagesCreate",
            display_name="Chat Message Append",
            category="api node/text/OpenAI",
            description="Append one message to an existing messages JSON array (chain multiple of these).",
            inputs=[
                IO.String.Input(
                    "messages_json",
                    default="[]",
                    multiline=True,
                    socketless=False,
                    optional=True,
                    force_input=True,
                    tooltip="Existing messages JSON array; leave as [] to start a new one.",
                ),
                IO.Combo.Input(
                    "role",
                    options=["user", "system", "assistant", "tool"],
                    default="user",
                    tooltip="Role for this message.",
                ),
                IO.String.Input(
                    "content",
                    default="",
                    multiline=True,
                    socketless=True,
                    force_input=False,
                    tooltip="Message text content (prompt field).",
                ),
                IO.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Optional image tensor to include (encoded as data URI).",
                ),
            ],
            outputs=[
                IO.String.Output(
                    id="messages_json_out",
                    display_name="Messages JSON",
                    tooltip="JSON array of messages.",
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

        messages = list(json.loads(messages_json)) if messages_json else []

        content_parts: List[Dict[str, Any]] = []
        if content:
            content_parts.append({"type": "text", "text": content})

        if image is not None:
            data_uri = tensor_to_data_uri(image, mime_type="image/png")
            content_parts.append(
                {"type": "image_url", "image_url": {"url": data_uri, "detail": "auto"}}
            )

        message_content: Any = (
            content_parts
            if content_parts
            else ([{"type": "text", "text": content}] if content else [])
        )

        message: Dict[str, Any] = {"role": role, "content": message_content}
        if role == "tool":
            tc_id = ""
            for prev in reversed(messages):
                if prev.get("role") != "assistant":
                    continue
                tc_list = prev.get("tool_calls")
                if isinstance(tc_list, list) and tc_list:
                    last_call = tc_list[-1]
                    if isinstance(last_call, dict):
                        tc_id = last_call.get("id") or last_call.get("call_id") or ""
                    if tc_id:
                        break
            if not tc_id:
                raise ValueError(
                    "role=tool requires an assistant tool_call upstream; none found."
                )
            message["tool_call_id"] = tc_id

        messages.append(message)
        return IO.NodeOutput(json.dumps(messages))
