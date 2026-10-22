import json
import math
from functools import cache
from itertools import product
from string import ascii_lowercase
from typing import Annotated, Any, ClassVar

import nodes
import PIL
from comfy_api.latest import IO
from PIL import Image
from pydantic import Field, TypeAdapter


class Eval(IO.ComfyNode):
    output_count: ClassVar[int] = 1

    @classmethod
    def define_schema(cls) -> IO.Schema:
        any_type = IO.Custom("*")
        return IO.Schema(
            node_id=cls.__name__,
            display_name="Eval",
            is_deprecated=cls.__name__ != "Eval",
            description="Evaluate Python with inputs a, b, c, etc. Includes math, json, PIL, and Image.",
            inputs=[
                IO.String.Input(
                    "expression",
                    default="a",
                    multiline=True,
                    tooltip="Python expression. Multiple outputs require a tuple or list of matching length.",
                ),
                IO.Autogrow.Input(
                    "values",
                    template=IO.Autogrow.TemplateNames(
                        input=any_type.Input("value", optional=True),
                        names=[*ascii_lowercase, *map("".join, product(ascii_lowercase, repeat=2))][:100],
                        min=1,
                    ),
                ),
            ],
            outputs=[
                any_type.Output(
                    id="result" if index == 0 else f"result_{index + 1}",
                    display_name="Result" if cls.output_count == 1 else f"Result {index + 1}",
                )
                for index in range(cls.output_count)
            ],
        )

    @classmethod
    def execute(cls, expression: str = "a", values: dict[str, Any] | None = None) -> IO.NodeOutput:
        result = eval(expression, {
            "__builtins__": __builtins__, "math": math, "json": json,
            "PIL": PIL, "Image": Image, "a": None, **(values or {}),
        })
        outputs = (result,) if cls.output_count == 1 else TypeAdapter(
            Annotated[list[Any] | tuple[Any, ...], Field(min_length=cls.output_count, max_length=cls.output_count)]
        ).validate_python(result, strict=True)
        return IO.NodeOutput(*outputs)


@cache
def register_eval_outputs(count: int) -> str:
    type_id = f"Eval_{count}"
    nodes.NODE_CLASS_MAPPINGS[type_id] = type(type_id, (IO.ComfyNode,), {
        "output_count": count,
        "define_schema": classmethod(Eval.define_schema.__func__),
        "execute": classmethod(Eval.execute.__func__),
    })
    return type_id
