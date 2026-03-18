import re

from comfy_api.latest import IO

_DEFAULT_DIMENSIONS = "768x1344, 832x1216, 896x1152, 1024x1024, 1152x896, 1216x832, 1344x768"


def _parse_dimensions(dimensions_str: str) -> list[tuple[int, int]]:
    pattern = re.compile(r"(\d+)\s*x\s*(\d+)", re.IGNORECASE)
    return [
        (int(m.group(1)), int(m.group(2)))
        for dim in re.split(r"\s*,\s*", dimensions_str.strip())
        if dim and (m := pattern.match(dim))
    ]


def _preset_options(dimensions_str: str) -> list[str]:
    return [f"{w}x{h}" for w, h in _parse_dimensions(dimensions_str)]


class DimensionsPresetPicker(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        defaults = _preset_options(_DEFAULT_DIMENSIONS)
        return IO.Schema(
            node_id="DimensionsPresetPicker",
            display_name="Dimensions Preset Picker",
            category="image/transform",
            description="Output width and height from a comma-separated XxY list; preset combobox is driven by the dimensions textbox.",
            inputs=[
                IO.String.Input(
                    "dimensions",
                    default=_DEFAULT_DIMENSIONS,
                    tooltip="Comma-separated list of dimensions in XxY format.",
                    multiline=True,
                ),
                IO.Combo.Input(
                    "preset",
                    options=defaults,
                    default=defaults[0] if defaults else "1024x1024",
                    tooltip="Chosen dimension preset from the list above.",
                ),
            ],
            outputs=[
                IO.Int.Output(
                    id="width",
                    display_name="Width",
                    tooltip="Width of the selected preset.",
                ),
                IO.Int.Output(
                    id="height",
                    display_name="Height",
                    tooltip="Height of the selected preset.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        dimensions: str = _DEFAULT_DIMENSIONS,
        preset: str = "768x1344",
    ) -> IO.NodeOutput:
        parsed = _parse_dimensions(preset)
        if not parsed:
            raise ValueError(f"Preset is not a valid dimension: {preset}")
        w, h = parsed[0]
        return IO.NodeOutput(int(w), int(h))
