import json
import re
from typing import Literal, TypedDict

from comfy_api.latest import IO, _io


class Parameter(TypedDict):
    name: str
    type: Literal["string", "integer", "number", "boolean"]
    description: str
    required: bool


class AddTool(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="BasencAddTool",
            display_name="Add Tool",
            inputs=[
                IO.String.Input("name", default="my_tool"),
                IO.String.Input("description", default="", multiline=True),
                _io._DynamicGroup.Input("parameters", group_name="Parameter", template=[
                    IO.String.Input("name", default=""),
                    IO.Combo.Input("type", options=["string", "integer", "number", "boolean"]),
                    IO.String.Input("description", default=""),
                    IO.Boolean.Input("required", default=True),
                ]),
                IO.String.Input("previous", optional=True, force_input=True,
                                tooltip="Connect Tools JSON from another Add Tool node."),
            ],
            outputs=[IO.String.Output(id="tools_json", display_name="Tools JSON")],
        )

    @classmethod
    def execute(cls, name: str, description: str, parameters: list[Parameter],
                previous: str = "[]") -> IO.NodeOutput:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise ValueError("Use 1–64 letters, digits, underscores, or hyphens for the tool name.")
        tools = json.loads(previous)
        if not isinstance(tools, list):
            raise ValueError("The previous Tools JSON input must contain a list.")
        properties: dict[str, dict[str, str]] = {}
        required: list[str] = []
        for row in parameters:
            if row["name"] is not None:  # ComfyUI pads removed rows with None fields.
                key = row["name"].strip()
                if not key or key in properties:
                    raise ValueError("Each parameter needs a unique, nonempty name.")
                properties[key] = {"type": row["type"], "description": row["description"]}
                if row["required"]:
                    required.append(key)
        tools.append({"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        }})
        return IO.NodeOutput(json.dumps(tools))
