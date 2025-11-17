from comfy_api.latest import IO


class Eval(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        any_type = IO.Custom("*")
        return IO.Schema(
            node_id="Eval",
            display_name="Eval",
            category="util",
            description="Evaluate a Python expression with the incoming value available as `x`.",
            inputs=[
                IO.String.Input(
                    "expression",
                    default="value",
                    multiline=False,
                    tooltip="Python expression. `x` refer to the input.",
                ),
                any_type.Input(
                    "value",
                    display_name="Value",
                    optional=True,
                    tooltip="Any input accessible as `x` inside the expression.",
                ),
            ],
            outputs=[
                any_type.Output(
                    id="result",
                    display_name="Result",
                    tooltip="Result of the evaluated expression.",
                ),
            ],
        )

    @classmethod
    def execute(cls, expression: str = "value", value=None) -> IO.NodeOutput:
        env = {
            "x": value,
            "math": __import__("math"),
            "json": __import__("json"),
        }
        try:
            result = eval(expression, {"__builtins__": __builtins__}, env)  # noqa: S307
        except Exception as exc:
            raise RuntimeError(f"Eval error: {exc}") from exc
        return IO.NodeOutput(result)
