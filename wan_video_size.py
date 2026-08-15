from typing import ClassVar, Literal

import torch
from comfy_api.latest import IO

ResolutionPreset = Literal["480p", "544p", "720p", "1080p"]
AspectRatio = Literal["auto", "16:9", "9:16", "1:1", "4:3", "3:4"]


class WanVideoSize(IO.ComfyNode):
    _ASPECT_RATIOS: ClassVar[dict[str, float]] = {
        "16:9": 16 / 9,
        "4:3": 4 / 3,
        "1:1": 1,
        "3:4": 3 / 4,
        "9:16": 9 / 16,
    }
    _SIZE_TABLE: ClassVar[dict[ResolutionPreset, dict[str, tuple[int, int]]]] = {
        "480p": {
            "16:9": (864, 480),
            "9:16": (480, 864),
            "1:1": (480, 480),
            "4:3": (640, 480),
            "3:4": (480, 640),
        },
        "544p": {
            "16:9": (960, 544),
            "9:16": (544, 960),
            "1:1": (544, 544),
            "4:3": (736, 544),
            "3:4": (544, 736),
        },
        "720p": {
            "16:9": (1280, 736),
            "9:16": (736, 1280),
            "1:1": (736, 736),
            "4:3": (960, 736),
            "3:4": (736, 960),
        },
        "1080p": {
            "16:9": (1920, 1088),
            "9:16": (1088, 1920),
            "1:1": (1088, 1088),
            "4:3": (1440, 1088),
            "3:4": (1088, 1440),
        },
    }

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="WanVideoSize",
            display_name="Wan Video Size",
            category="util/video",
            description="Width/height for Wan video, based on image aspect ratio.",
            inputs=[
                IO.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Reference image used for aspect detection in auto mode.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=list(cls._SIZE_TABLE.keys()),
                    default="720p",
                    tooltip="Resolution preset.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["auto", "16:9", "9:16", "1:1", "4:3", "3:4"],
                    default="auto",
                    tooltip="Use the nearest image aspect ratio in auto, or force a ratio.",
                ),
            ],
            outputs=[
                IO.Int.Output(
                    id="width",
                    display_name="Width",
                    tooltip="Width for the chosen preset/orientation.",
                ),
                IO.Int.Output(
                    id="height",
                    display_name="Height",
                    tooltip="Height for the chosen preset/orientation.",
                ),
            ],
        )

    @classmethod
    def _aspect_ratio_from_image(cls, image: torch.Tensor) -> str:
        h = int(image.shape[-3])
        w = int(image.shape[-2])
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image dimensions: {w}x{h}")
        return min(
            cls._ASPECT_RATIOS, key=lambda name: abs(w / h - cls._ASPECT_RATIOS[name])
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor | None = None,
        resolution: ResolutionPreset = "720p",
        aspect_ratio: AspectRatio = "auto",
    ) -> IO.NodeOutput:
        if aspect_ratio == "auto":
            if image is None:
                raise ValueError("image is required when aspect_ratio is auto.")
            aspect_ratio = cls._aspect_ratio_from_image(image)

        resolution_sizes = cls._SIZE_TABLE.get(resolution)
        if resolution_sizes is None:
            raise ValueError(f"Unknown resolution preset: {resolution}")

        if aspect_ratio not in resolution_sizes:
            raise ValueError(f"Unknown aspect ratio: {aspect_ratio}")

        return IO.NodeOutput(*resolution_sizes[aspect_ratio])
