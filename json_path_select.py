import json
from typing import Any

import jmespath

from comfy_api.latest import IO


class JSONPathSelect(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="JMESPathSelect",
            display_name="JSON Path Select",
            category="util/json",
            description="Extract a value from JSON using JMESPath.",
            inputs=[
                IO.String.Input(
                    "json_text",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="JSON text to query.",
                ),
                IO.String.Input(
                    "path",
                    default="",
                    tooltip="JMESPath (e.g. choices[0].message.tool_calls[0].id)",
                ),
            ],
            outputs=[
                IO.String.Output(
                    id="value",
                    display_name="Value",
                    tooltip="Extracted value (stringified if not a string). Empty if not found.",
                )
            ],
        )

    @classmethod
    def execute(cls, json_text: str = "", path: str = "") -> IO.NodeOutput:
        if not json_text or not path:
            return IO.NodeOutput("")

        data: Any = json.loads(json_text)

        try:
            current = jmespath.search(path, data)
        except Exception:
            return IO.NodeOutput("")

        if isinstance(current, (str, int, float, bool)):
            return IO.NodeOutput(str(current))

        try:
            return IO.NodeOutput(json.dumps(current, indent=2))
        finally:
            return IO.NodeOutput("")
