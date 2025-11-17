import os
from pathlib import Path
from typing import List, Optional

from dotenv import dotenv_values

from comfy_api.latest import io


class EnvVarNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        env_choices = [""] + cls._available_env_keys()
        return io.Schema(
            node_id="EnvVarNode",
            display_name="Environment Variable",
            category="utils",
            inputs=[
                io.Combo.Input("env_key", default="", options=env_choices),
                io.String.Input(
                    "default_value",
                    default="",
                    optional=True,
                    placeholder="Optional fallback value",
                ),
                io.Boolean.Input(
                    "use_fallback_when_empty",
                    default=True,
                    label_off="Only when missing",
                    label_on="Also when empty",
                    optional=True,
                ),
                io.Boolean.Input(
                    "error_when_missing",
                    default=False,
                    label_off="Allow fallback/empty",
                    label_on="Raise error",
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
        use_fallback_when_empty: bool = True,
        error_when_missing: bool = False,
    ) -> io.NodeOutput:
        if not env_key:
            if error_when_missing:
                raise ValueError("No environment variable key provided.")
            return io.NodeOutput("")

        dotenv_keys = cls._dotenv_keys()
        resolved = cls._dotenv_value_for(env_key)
        env_value = os.environ.get(env_key, "")

        if env_value:
            resolved = env_value

        if not resolved and (default_value or use_fallback_when_empty):
            if (
                error_when_missing
                and env_key not in os.environ
                and env_key not in dotenv_keys
            ):
                raise ValueError(f"Environment variable '{env_key}' is not set.")
            if use_fallback_when_empty or env_key not in os.environ:
                resolved = default_value
        elif (
            not resolved
            and error_when_missing
            and env_key not in os.environ
            and env_key not in dotenv_keys
        ):
            raise ValueError(f"Environment variable '{env_key}' is not set.")

        return io.NodeOutput(resolved)

    @classmethod
    def _node_root(cls) -> Path:
        return Path(__file__).resolve().parent

    @classmethod
    def _dotenv_path(cls) -> Optional[Path]:
        # Check in ComfyUI root directory
        comfy_root = cls._node_root().parent.parent
        env_file = comfy_root / ".env"
        return env_file if env_file.exists() else None

    @classmethod
    def _dotenv_keys(cls) -> List[str]:
        dotenv_path = cls._dotenv_path()
        if not dotenv_path:
            return []
        return list(dotenv_values(dotenv_path).keys())

    @classmethod
    def _available_env_keys(cls) -> List[str]:
        keys: set[str] = set(cls._dotenv_keys())

        for key in os.environ.keys():
            if key.startswith("COMFYUI_"):
                keys.add(key)

        return sorted(keys)

    @classmethod
    def _dotenv_value_for(cls, env_key: str) -> str:
        if not env_key:
            return ""
        dotenv_path = cls._dotenv_path()
        if not dotenv_path:
            return ""
        return dotenv_values(dotenv_path).get(env_key, "")
