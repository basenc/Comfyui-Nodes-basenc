import os

from comfy_api.latest import io
from dotenv import dotenv_values, find_dotenv


def available_env_keys() -> list[str]:
    keys: set[str] = set(os.environ.keys())
    keys.update(dotenv_values(find_dotenv()).keys())
    return sorted(keys)


class EnvVarNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EnvVarNode",
            display_name="Environment Variable",

            inputs=[
                io.Combo.Input(
                    "env_key",
                    default="",
                    options=["", *available_env_keys()],
                ),
                io.String.Input(
                    "default_value",
                    default="",
                    optional=True,
                    placeholder="Optional fallback value",
                ),
                io.Combo.Input(
                    "missing_value",
                    options=["Also when empty", "Only when missing", "Raise error"],
                    default="Also when empty",
                    optional=True,
                ),
            ],
            outputs=[io.String.Output(display_name="value")],
        )

    @classmethod
    def execute(
        cls,
        env_key: str,
        default_value: str = "",
        missing_value: str = "Also when empty",
    ) -> io.NodeOutput:
        value = os.environ.get(env_key, dotenv_values(find_dotenv()).get(env_key)) if env_key else None
        if missing_value == "Raise error" and value is None:
            raise ValueError(f"Environment variable '{env_key}' is not set.")
        return io.NodeOutput(
            default_value
            if value is None or (missing_value == "Also when empty" and not value)
            else value
        )


