import ctypes
import json
import sys
from pathlib import Path

from llama_cpp import Llama, llama_cpp
from llama_cpp.llama_chat_format import MTMDChatHandler


def _apply_loras(model: Llama, loras: list[list[str | float]]) -> None:
    adapters = []
    scales = []
    for path, strength in loras:
        adapter = llama_cpp.llama_adapter_lora_init(
            model.model,
            str(path).encode("utf-8"),
        )
        if adapter is None:
            raise RuntimeError(f"Failed to load LoRA: {path}")
        if llama_cpp.llama_adapter_get_alora_n_invocation_tokens(adapter):
            llama_cpp.llama_adapter_lora_free(adapter)
            raise ValueError("aLoRA adapters are not supported.")
        adapters.append(adapter)
        scales.append(float(strength))
    if (
        error := (
            llama_cpp.llama_set_adapters_lora(
                model.ctx,
                (llama_cpp.llama_adapter_lora_p_ctypes * len(adapters))(*adapters),
                len(adapters),
                (ctypes.c_float * len(scales))(*scales),
            )
            if adapters
            else llama_cpp.llama_set_adapters_lora(model.ctx, None, 0, None)
        )
    ) != 0:
        raise RuntimeError(f"Failed to apply LoRAs: error {error}")


def _load_model(request):
    mtmd = (
        MTMDChatHandler(
            clip_model_path=request["mmproj_path"],
            verbose=False,
            use_gpu=True,
        )
        if request["mmproj_path"] is not None
        else None
    )
    model = Llama(
        model_path=request["model_path"],
        chat_format=("chat_template.default" if mtmd is None else None),
        chat_handler=mtmd,
        n_ctx=request["n_ctx"],
        n_batch=request["n_batch"],
        n_gpu_layers=request["gpu_layers"],
        flash_attn=request["flash_attn"],
        verbose=False,
    )
    handler = mtmd or model._chat_handlers.get("chat_template.default")
    if handler is None:
        model.close()
        raise ValueError(
            "The model GGUF does not contain tokenizer.chat_template metadata."
        )
    return model, handler, mtmd


def _complete(model, handler, request):
    _apply_loras(model, request["loras"])
    return handler(
        llama=model,
        messages=request["messages"],
        tools=request["tools"] or None,
        tool_choice=request["tool_choice"],
        temperature=request["temperature"],
        max_tokens=request["max_tokens"],
        stream=False,
        enable_thinking=request["thinking"],
    )


def _run(request):
    model, handler, mtmd = _load_model(request)
    try:
        return _complete(model, handler, request)
    finally:
        if mtmd is not None:
            mtmd._exit_stack.close()
        model.close()


def _read_request(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_response(path: Path, response) -> None:
    path.write_text(json.dumps(response), encoding="utf-8")


def _main(request_path: Path, response_path: Path) -> None:
    _write_response(response_path, _run(_read_request(request_path)))


_main(Path(sys.argv[1]), Path(sys.argv[2]))
