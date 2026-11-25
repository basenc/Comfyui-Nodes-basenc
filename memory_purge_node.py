import gc
from typing import Any

import comfy.model_management
from comfy_api.latest import IO

_ANY = IO.Custom("*")

class PurgeVRAM(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="BasencPurgeVRAM",
            display_name="Purge VRAM",
            description="Unload all models and empty the accelerator memory caches.",
            inputs=[
                _ANY.Input(
                    "*",
                    optional=True,
                    tooltip="Optional value to pass through after the purge.",
                ),
            ],
            outputs=[_ANY.Output(id="anything", display_name="*")],
        )

    @classmethod
    def execute(cls, **inputs: Any) -> IO.NodeOutput:
        comfy.model_management.unload_all_models()
        # Release model and tensor references before emptying the caches.
        gc.collect()
        comfy.model_management.soft_empty_cache(force=True)
        return IO.NodeOutput(inputs.get("*"))
