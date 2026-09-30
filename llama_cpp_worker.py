import ctypes
import json
import sys
from importlib.metadata import distribution
from typing import TypedDict


type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]


class CompletionRequest(TypedDict):
    model_path: str
    mmproj_path: str | None
    messages: list[dict[str, JsonValue]]
    tools: list[dict[str, JsonValue]]
    tool_choice: str | None
    thinking: bool
    chat_template_kwargs: dict[str, JsonValue]
    temperature: float
    max_tokens: int | None
    n_ctx: int
    n_batch: int
    gpu_layers: int
    verbose: bool


def main() -> None:
    lib_dir = distribution("llama-cpp-python").locate_file("llama_cpp/lib")
    ctypes.CDLL(str(lib_dir / "libggml-base.so"), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(lib_dir / "libggml.so"), mode=ctypes.RTLD_GLOBAL)

    from llama_cpp import Llama, llama_cpp
    from llama_cpp.llama_chat_format import MTMDChatHandler

    load = llama_cpp._lib.ggml_backend_load_all_from_path
    load.argtypes = [ctypes.c_char_p]
    load.restype = None
    load(str(llama_cpp._base_path).encode())

    request: CompletionRequest = json.load(sys.stdin)
    mtmd = (
        MTMDChatHandler(
            clip_model_path=request["mmproj_path"],
            verbose=request["verbose"],
            use_gpu=True,
        )
        if request["mmproj_path"] is not None
        else None
    )
    model = Llama(
        model_path=request["model_path"],
        chat_format="chat_template.default" if mtmd is None else None,
        chat_handler=mtmd,
        n_ctx=request["n_ctx"],
        n_batch=request["n_batch"],
        n_gpu_layers=request["gpu_layers"],
        verbose=request["verbose"],
    )
    try:
        if not model.metadata.get("tokenizer.chat_template"):
            raise ValueError("The model GGUF does not contain tokenizer.chat_template metadata.")
        handler = mtmd or model._chat_handlers["chat_template.default"]
        json.dump(
            handler(
                llama=model,
                messages=request["messages"],
                tools=request["tools"] or None,
                tool_choice=request["tool_choice"],
                temperature=request["temperature"],
                max_tokens=request["max_tokens"],
                stream=False,
                enable_thinking=request["thinking"],
                **request["chat_template_kwargs"],
            ),
            sys.stdout,
        )
    finally:
        model.close()


if __name__ == "__main__":
    main()
