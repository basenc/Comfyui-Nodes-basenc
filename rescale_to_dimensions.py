import re

import torch
from comfy_api.latest import IO
from torch.nn.functional import interpolate


class RescaleToDimensions(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="RescaleToDimensions",
            display_name="Rescale To Dimensions",
            category="image/transform",
            description="Rescale image to nearest dimension from comma-separated XxY list, cropping to fit height.",
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="Input image to rescale.",
                ),
                IO.String.Input(
                    "dimensions",
                    default="768x1344, 832x1216, 896x1152, 1024x1024, 1152x896, 1216x832, 1344x768",
                    tooltip="Comma-separated list of dimensions in XxY format.",
                    multiline=True,
                ),
            ],
            outputs=[
                IO.Image.Output(
                    id="output_image",
                    display_name="Image",
                    tooltip="Rescaled and cropped image.",
                ),
                IO.Int.Output(
                    id="width",
                    display_name="Width",
                    tooltip="Final width.",
                ),
                IO.Int.Output(
                    id="height",
                    display_name="Height",
                    tooltip="Final height.",
                ),
            ],
        )

    @classmethod
    def _parse_dimensions(cls, dimensions_str: str) -> list[tuple[int, int]]:
        pattern = re.compile(r"(\d+)\s*x\s*(\d+)", re.IGNORECASE)
        return [
            (int(m.group(1)), int(m.group(2)))
            for dim in re.split(r"\s*,\s*", dimensions_str.strip())
            if dim and (m := pattern.match(dim))
        ]

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
        image: torch.Tensor,
        dimensions: str = "768x1344, 832x1216, 896x1152, 1024x1024, 1152x896, 1216x832, 1344x768",
    ) -> IO.NodeOutput:
        if image is None:
            raise ValueError("image is required")

        targets = cls._parse_dimensions(dimensions)
        if not targets:
            raise ValueError(f"No valid dimensions parsed from: {dimensions}")

        b, h, w, c = image.shape
        target_w, target_h = cls._find_best_dimension(w, h, targets)

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
        )

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
