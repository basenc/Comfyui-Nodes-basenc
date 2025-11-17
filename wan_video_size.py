from typing import Literal

import torch

from comfy_api.latest import IO

ResolutionPreset = Literal["480p", "720p", "1080p"]
Orientation = Literal["auto", "landscape", "portrait", "square"]


class WanVideoSize(IO.ComfyNode):
    _SIZE_TABLE: dict[ResolutionPreset, dict[str, tuple[int, int]]] = {
        "480p": {
            "landscape": (640, 480),
            "portrait": (480, 640),
            "square": (480, 480),
        },
        "720p": {
            "landscape": (1280, 720),
            "portrait": (720, 1280),
            "square": (720, 720),
        },
        "1080p": {
            "landscape": (1920, 1088),
            "portrait": (1088, 1920),
            "square": (1088, 1088),
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
                    tooltip="Reference image used for aspect detection in auto mode.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=list(cls._SIZE_TABLE.keys()),
                    default="720p",
                    tooltip="Resolution preset.",
                ),
                IO.Combo.Input(
                    "orientation",
                    options=["auto", "landscape", "portrait", "square"],
                    default="auto",
                    tooltip="Use image aspect in auto, or force an orientation.",
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
    def _orientation_from_image(
        cls, image: torch.Tensor, threshold: float = 0.1
    ) -> str:
        h = int(image.shape[-3])
        w = int(image.shape[-2])
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image dimensions: {w}x{h}")
        min_dim = min(w, h)
        diff = abs(w - h)
        if min_dim > 0 and diff / min_dim < threshold:
            return "square"
        if w > h:
            return "landscape"
        if h > w:
            return "portrait"
        return "square"

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        resolution: ResolutionPreset = "720p",
        orientation: Orientation = "auto",
    ) -> IO.NodeOutput:
        if image is None:
            raise ValueError("image is required.")

        if orientation == "auto":
            orientation = cls._orientation_from_image(image)

        resolution_sizes = cls._SIZE_TABLE.get(resolution)
        if resolution_sizes is None:
            raise ValueError(f"Unknown resolution preset: {resolution}")

        if orientation not in resolution_sizes:
            raise ValueError(f"Unknown orientation: {orientation}")

        w, h = resolution_sizes[orientation]
        return IO.NodeOutput(int(w), int(h))
