import asyncio
import contextlib
import json
from functools import cache
from pathlib import Path
from typing import Literal

import comfy.model_management
import folder_paths
import nodes
from comfy_api.latest import IO
from comfy_api_nodes.util.conversions import tensor_to_data_uri
from torch import Tensor
from typesafe_sdk import AsyncTypeSafeClient, Choice

from .llama_cpp_backend import _MODEL_FOLDER, _llama_server, _model_options
from .secret_store import resolve_secret


CHOICES = IO.Custom("BASENC_CHOICES")


class ChoiceCriteria(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="ChoiceCriteria",
            display_name="Choice",
            inputs=[
                IO.String.Input("name", default="yes"),
                IO.String.Input("description", default="", multiline=True),
                CHOICES.Input("previous", optional=True),
            ],
            outputs=[CHOICES.Output(id="choices", display_name="Choices")],
        )

    @classmethod
    def execute(cls, name: str, description: str, previous: str | None = None) -> IO.NodeOutput:
        criteria = json.loads(previous) if previous else {}
        if not name or name in criteria or len(criteria) >= 255:
            raise ValueError("Choice names must be unique and nonempty (maximum 255).")
        criteria[name] = description or None
        return IO.NodeOutput(json.dumps(criteria))


def _choice_outputs(names: list[str]) -> list[IO.String.Output | IO.Boolean.Output]:
    return [
        IO.String.Output(id="choice", display_name="Choice"),
        IO.String.Output(id="probabilities_json", display_name="Probabilities JSON"),
        *(IO.Boolean.Output(id=f"choice_{i}", display_name=name) for i, name in enumerate(names)),
    ]


def _choice_schema(
    node_id: str,
    display_name: str,
    inputs: list[IO.String.Input | IO.Combo.Input],
) -> IO.Schema:
    return IO.Schema(
        node_id=node_id,
        display_name=display_name,
        inputs=[
            *inputs,
            IO.String.Input("state", default="", multiline=True),
            IO.String.Input("instructions", default="Which option best matches the state?", multiline=True),
            CHOICES.Input("choices", tooltip="Connect the Choices output of a Choice node."),
            IO.Autogrow.Input(
                "images",
                template=IO.Autogrow.TemplatePrefix(
                    input=IO.Image.Input("image", tooltip="Requires an image-capable model and API."),
                    prefix="image",
                    min=0,
                ),
            ),
        ],
        outputs=_choice_outputs([]),
    )


async def _request_choice(
    api_base: str,
    api_key: str | None,
    model: str,
    state: str,
    question: Choice,
    images: dict[str, Tensor] | None = None,
) -> IO.NodeOutput:
    comfy.model_management.throw_exception_if_processing_interrupted()
    async with (
        AsyncTypeSafeClient(api_key=api_key, base_url=api_base, model=model) as client,
        contextlib.AsyncExitStack() as cleanup,
    ):
        request = asyncio.create_task(
            client.system_one(
                state=state,
                questions={"choice": question},
                extra_body={
                    "images": [
                        tensor_to_data_uri(image, mime_type="image/png")
                        for batch in images.values()
                        for image in batch.split(1)
                    ]
                } if images else None,
            )
        )
        cleanup.push_async_callback(asyncio.gather, request, return_exceptions=True)
        cleanup.callback(request.cancel)
        while not request.done():
            await asyncio.wait({request}, timeout=0.25)
            comfy.model_management.throw_exception_if_processing_interrupted()
        answer = (await request).choices["choice"]
    return IO.NodeOutput(
        answer.choice,
        json.dumps(answer.probabilities, indent=2),
        *(answer.choice == name for name in question.criteria),
    )


class TypeSafeChoice(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _choice_schema(
            "TypeSafeChoice",
            "TypeSafe Choice",
            [
                IO.String.Input("api_base", default="https://api.typesafe.ai", optional=True),
                IO.String.Input("api_key", default="", optional=True),
                IO.String.Input("model", default="jev-latest", optional=True),
            ],
        )

    @classmethod
    async def execute(
        cls,
        state: str,
        instructions: str,
        choices: str,
        api_base: str = "https://api.typesafe.ai",
        api_key: str = "",
        model: str = "jev-latest",
        images: dict[str, Tensor] | None = None,
    ) -> IO.NodeOutput:
        question = Choice.model_validate({"instructions": instructions, "criteria": json.loads(choices)})
        return await _request_choice(
            api_base, resolve_secret(api_key) if api_key else None, model, state, question, images
        )


class LlamaCppChoice(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _choice_schema(
            "LlamaCppChoice",
            "llama.cpp Choice",
            [
                IO.Combo.Input(
                    "model",
                    options=_model_options(mmproj=False),
                    tooltip="Decision-model GGUF. Requires llama-server with /v1/systemone support.",
                ),
                IO.Combo.Input(
                    "mmproj",
                    options=["None", *_model_options(mmproj=True)],
                    default="None",
                    optional=True,
                    tooltip="Vision projector for image input. None runs text-only decisions.",
                ),
                IO.String.Input(
                    "n_ctx",
                    default="auto",
                    optional=True,
                    tooltip="Context size: auto fits available memory, or enter a token count.",
                ),
                IO.String.Input(
                    "gpu_layers",
                    default="auto",
                    optional=True,
                    tooltip="GPU layers: auto, all, or a number. Memory fitting is enabled.",
                ),
                IO.String.Input(
                    "llama_server_args",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Extra llama-server flags. Quoting is supported; node flags take priority.",
                ),
            ],
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        state: str,
        instructions: str,
        choices: str,
        n_ctx: str = "auto",
        gpu_layers: str = "auto",
        llama_server_args: str = "",
        mmproj: str = "None",
        images: dict[str, Tensor] | None = None,
    ) -> IO.NodeOutput:
        question = Choice.model_validate({"instructions": instructions, "criteria": json.loads(choices)})
        model_path = folder_paths.get_full_path_or_raise(_MODEL_FOLDER, model)
        async with _llama_server(
            model_path=model_path,
            mmproj_path=(
                None if mmproj == "None" else folder_paths.get_full_path_or_raise(_MODEL_FOLDER, mmproj)
            ),
            n_ctx=n_ctx,
            gpu_layers=gpu_layers,
            llama_server_args=llama_server_args,
        ) as api_base:
            return await _request_choice(api_base, "sk-noop", Path(model_path).name, state, question, images)


@cache
def register_choice_outputs(node_id: Literal["TypeSafeChoice", "LlamaCppChoice"], count: int) -> str:
    base = {"TypeSafeChoice": TypeSafeChoice, "LlamaCppChoice": LlamaCppChoice}[node_id]
    type_id = f"{node_id}_{count}"

    class ChoiceOutputs(IO.ComfyNode):
        @classmethod
        def define_schema(cls) -> IO.Schema:
            schema = base.define_schema()
            schema.node_id = type_id
            schema.outputs = _choice_outputs([f"option_{i + 1}" for i in range(count)])
            schema.is_deprecated = True
            return schema

        execute = classmethod(base.execute.__func__)

    nodes.NODE_CLASS_MAPPINGS[type_id] = ChoiceOutputs
    return type_id
