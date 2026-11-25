import json

import ai
from comfy_api.latest import IO
from comfy_api_nodes.util.conversions import (
    audio_to_base64_string,
    tensor_to_data_uri,
    video_to_base64_string,
)
from torch import Tensor


type Messages = list[ai.messages.Message]

MESSAGES = IO.Custom("BASENC_MESSAGES")


type _Json = str | int | float | bool | None | list[_Json] | dict[str, _Json]


def _console_value(value: _Json | bytes) -> _Json:
    if isinstance(value, dict):
        if value.get("kind") == "file":
            return "[file omitted]"
        return {
            key: _console_value(item) for key, item in value.items()
            if key not in {"id", "ids", "trace", "traces", "traceback", "provider_metadata"}
            and not key.endswith(("_id", "_ids"))
        }
    if isinstance(value, (list, tuple)):
        return [_console_value(item) for item in value]
    if isinstance(value, bytes) or (isinstance(value, str) and value.startswith("data:")):
        return "[file omitted]"
    return value


def print_message(message: ai.messages.Message) -> None:
    fields = {"kind", "text", "tool_name", "tool_args", "result", "is_error",
              "hook_type", "status"}
    parts = []
    for part in message.parts:
        if part.kind == "file":
            continue
        view = part.model_dump(mode="python", include=fields, exclude_none=True)
        if args := view.get("tool_args"):
            try:
                view["tool_args"] = json.loads(args)
            except json.JSONDecodeError:
                pass  # Built-in tools can use plain text instead of JSON.
        parts.append(_console_value(view))
    output: dict[str, _Json] = {"role": message.role, "parts": parts}
    if message.usage is not None:
        output["usage"] = message.usage.model_dump(exclude={"raw"}, exclude_none=True)
    print(json.dumps(output, indent=2, default=lambda _: "[non-JSON value omitted]"), flush=True)


class AddMessage(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AddMessage",
            display_name="Add Message",
            inputs=[
                MESSAGES.Input("messages", optional=True),
                IO.Combo.Input(
                    "role", options=["system", "user", "assistant", "tool"], default="user"
                ),
                IO.String.Input(
                    "content", default="", multiline=True, socketless=True, force_input=False
                ),
                IO.Combo.Input(
                    "strategy", options=["merge consecutive roles", "append"], default="merge consecutive roles"
                ),
                IO.Autogrow.Input(
                    "media",
                    template=IO.Autogrow.TemplatePrefix(
                        input=IO.MultiType.Input(
                            "media", types=[IO.Image, IO.Video, IO.Audio]
                        ),
                        prefix="media",
                        min=0,
                    ),
                ),
            ],
            outputs=[MESSAGES.Output(id="messages", display_name="Messages")],
        )

    @classmethod
    def execute(
        cls,
        messages: Messages | None = None,
        role: str = "user",
        content: str = "",
        strategy: str = "merge consecutive roles",
        media: IO.Autogrow.Type | None = None,
    ) -> IO.NodeOutput:
        messages = list(messages or [])
        if role == "tool":
            call = next(
                (part for message in reversed(messages) for part in reversed(message.tool_calls)),
                None,
            )
            if not call:
                raise ValueError("A tool message requires a preceding tool call.")
            message = ai.tool_message(
                tool_call_id=call.tool_call_id, result=content, tool_name=call.tool_name
            )
        else:
            parts: list[str | ai.messages.FilePart] = [content] if content else []
            for attachment in (media or {}).values():
                if attachment is None:
                    continue
                if isinstance(attachment, Tensor):
                    parts.append(
                        ai.file_part(
                            tensor_to_data_uri(attachment, mime_type="image/png"),
                            media_type="image/png",
                        )
                    )
                elif isinstance(attachment, dict):
                    parts.append(
                        ai.file_part(
                            f"data:audio/mp4;base64,{audio_to_base64_string(attachment)}",
                            media_type="audio/mp4",
                        )
                    )
                else:
                    parts.append(
                        ai.file_part(
                            f"data:video/mp4;base64,{video_to_base64_string(attachment)}",
                            media_type="video/mp4",
                        )
                    )
            message = ai.message(*parts, role=role)
        if strategy == "merge consecutive roles" and messages and messages[-1].role == role:
            messages[-1] = messages[-1].model_copy(
                update={"parts": [*messages[-1].parts, *message.parts]}
            )
        else:
            messages.append(message)
        return IO.NodeOutput(messages)
