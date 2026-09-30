import re

import torch
from comfy_api.latest import IO
from torch.nn.functional import interpolate

_DEFAULT_DIMENSIONS = "768x1344, 832x1216, 896x1152, 1024x1024, 1152x896, 1216x832, 1344x768"


def _parse_dimensions(dimensions_str: str) -> list[tuple[int, int]]:
    pattern = re.compile(r"([1-9]\d*)\s*x\s*([1-9]\d*)", re.IGNORECASE)
    return [
        (int(match.group(1)), int(match.group(2)))
        for dimension in dimensions_str.split(",")
        if (match := pattern.fullmatch(dimension.strip()))
    ]


def _preset_options(dimensions_str: str) -> list[str]:
    return [f"{w}x{h}" for w, h in _parse_dimensions(dimensions_str)]


class ImageDimensions(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        defaults = _preset_options(_DEFAULT_DIMENSIONS)
        return IO.Schema(
            node_id="ImageDimensions",
            display_name="Image Dimensions",
            description="Select dimensions from a list and optionally resize an image to fit.",
            inputs=[
                IO.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Optional image to resize and crop to the selected dimensions.",
                ),
                IO.String.Input(
                    "dimensions",
                    default=_DEFAULT_DIMENSIONS,
                    tooltip="Comma-separated list of dimensions in XxY format.",
                    multiline=True,
                ),
                IO.Combo.Input(
                    "preset",
                    options=["Auto", *defaults],
                    default="Auto",
                    tooltip="Choose a preset, or Auto for the nearest fit to the image.",
                ),
            ],
            outputs=[
                IO.Image.Output(
                    id="output_image",
                    display_name="Image",
                    tooltip="Rescaled and cropped image, or None when no image is provided.",
                ),
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
    def validate_inputs(cls, dimensions: str, preset: str) -> bool | str:
        targets = _parse_dimensions(dimensions)
        if not targets:
            return f"No valid dimensions parsed from: {dimensions}"
        parsed = _parse_dimensions(preset)
        return preset == "Auto" or (len(parsed) == 1 and parsed[0] in targets) or f"Preset is not a valid dimension: {preset}"

    @classmethod
    def _find_best_dimension(
        cls, img_w: int, img_h: int, targets: list[tuple[int, int]]
    ) -> tuple[int, int]:
        img_aspect = img_w / img_h

        def score(target: tuple[int, int]) -> float:
            tw, th = target
            target_aspect = tw / th
            aspect_diff = abs(img_aspect - target_aspect)
            scale_factor = min(tw / img_w, th / img_h)
            return aspect_diff + abs(1.0 - scale_factor) * 0.1

        return min(targets, key=score)

    @classmethod
    def execute(
        cls,
        dimensions: str = _DEFAULT_DIMENSIONS,
        preset: str = "Auto",
        image: torch.Tensor | None = None,
    ) -> IO.NodeOutput:
        targets = _parse_dimensions(dimensions)
        if not targets:
            raise ValueError(f"No valid dimensions parsed from: {dimensions}")
        parsed = _parse_dimensions(preset)
        if preset != "Auto" and (len(parsed) != 1 or parsed[0] not in targets):
            raise ValueError(f"Preset is not a valid dimension: {preset}")

        if image is None:
            target_w, target_h = targets[0] if preset == "Auto" else parsed[0]
            return IO.NodeOutput(None, target_w, target_h)

        _, h, w, _ = image.shape
        target_w, target_h = cls._find_best_dimension(w, h, targets) if preset == "Auto" else parsed[0]
        scale = target_h / h
        scaled_w = int(w * scale)
        scaled_h = target_h

        image_permuted = image.permute(0, 3, 1, 2)
        scaled = interpolate(
            image_permuted,
            size=(scaled_h, scaled_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp_(0.0, 1.0)

        if scaled_w > target_w:
            crop_left = (scaled_w - target_w) // 2
            cropped = scaled[:, :, :, crop_left : crop_left + target_w]
        elif scaled_w < target_w:
            pad_left = (target_w - scaled_w) // 2
            pad_right = target_w - scaled_w - pad_left
            cropped = torch.nn.functional.pad(
                scaled, (pad_left, pad_right, 0, 0), mode="replicate"
            )
        else:
            cropped = scaled

        result = cropped.permute(0, 2, 3, 1)
        return IO.NodeOutput(result, target_w, target_h)
