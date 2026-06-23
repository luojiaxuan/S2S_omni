from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _safe_request_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180] or "request"


def stack_code_chunks(code_chunks: list[torch.Tensor]) -> np.ndarray:
    if not code_chunks:
        raise ValueError("no code chunks to stack")
    stacked = torch.stack([chunk.detach().to(dtype=torch.long) for chunk in code_chunks], dim=0)
    codes = stacked.transpose(0, 1).contiguous().cpu().numpy()
    return codes.astype(np.int16, copy=False)


class CapturingCode2WavScheduler:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
            Code2WavScheduler,
        )

        self._inner = Code2WavScheduler(*args, **kwargs)
        self._inner_on_streaming_new_request = self._inner.on_streaming_new_request
        self._inner_clear_stream_state = self._inner.clear_stream_state
        self._inner_abort = self._inner.abort
        self._inner_on_stream_done = self._inner.on_stream_done
        self._inner.on_streaming_new_request = self.on_streaming_new_request
        self._inner.clear_stream_state = self.clear_stream_state
        self._inner.abort = self.abort
        self._inner.on_stream_done = self.on_stream_done
        self._code_sidecar_paths: dict[str, Path] = {}
        self._saved_code_shapes: dict[str, list[int]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def start(self) -> None:
        self._inner.start()

    def stop(self) -> None:
        self._inner.stop()

    def abort(self, request_id: str) -> None:
        self._code_sidecar_paths.pop(request_id, None)
        self._saved_code_shapes.pop(request_id, None)
        self._inner_abort(request_id)

    def clear_stream_state(self, request_id: str) -> None:
        self._code_sidecar_paths.pop(request_id, None)
        self._saved_code_shapes.pop(request_id, None)
        self._inner_clear_stream_state(request_id)

    def on_streaming_new_request(self, request_id: str, payload: Any) -> None:
        self._inner_on_streaming_new_request(request_id, payload)
        metadata = getattr(getattr(payload, "request", None), "metadata", None)
        if not isinstance(metadata, dict):
            return
        code_path = metadata.get("code2wav_codes_path")
        code_dir = metadata.get("code2wav_codes_dir")
        if code_path:
            path = Path(str(code_path))
        elif code_dir:
            path = Path(str(code_dir)) / f"{_safe_request_name(request_id)}.npy"
        else:
            return
        self._code_sidecar_paths[request_id] = path

    def on_stream_chunk(self, request_id: str, chunk: Any) -> list[Any]:
        return self._inner.on_stream_chunk(request_id, chunk)

    def on_stream_done(self, request_id: str) -> list[Any]:
        sidecar_path = self._code_sidecar_paths.get(request_id)
        if sidecar_path is not None:
            self._save_codes(request_id, sidecar_path)
        messages = self._inner_on_stream_done(request_id)
        self._attach_sidecar_metadata(request_id, messages)
        return messages

    def is_streaming_payload(self, payload: Any) -> bool:
        return self._inner.is_streaming_payload(payload)

    def _save_codes(self, request_id: str, path: Path) -> None:
        code_chunks = self._inner._code_chunks.get(request_id, [])
        codes = stack_code_chunks(code_chunks)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, codes)
        self._saved_code_shapes[request_id] = [int(v) for v in codes.shape]

    def _attach_sidecar_metadata(self, request_id: str, messages: list[Any]) -> None:
        sidecar_path = self._code_sidecar_paths.get(request_id)
        if sidecar_path is None:
            return
        code_shape = self._saved_code_shapes.get(request_id)
        for message in messages:
            if getattr(message, "type", None) != "result":
                continue
            payload = getattr(message, "data", None)
            data = getattr(payload, "data", None)
            if isinstance(data, dict):
                data["codec_codes_path"] = str(sidecar_path)
                if code_shape is not None:
                    data["codec_shape"] = list(code_shape)


def create_capturing_code2wav_scheduler(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    gpu_id: int | None = None,
    stream_chunk_size: int = 10,
    left_context_size: int = 25,
) -> CapturingCode2WavScheduler:
    from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
        load_code2wav_model,
    )

    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    model = load_code2wav_model(model_path, device=device, dtype=dtype)
    return CapturingCode2WavScheduler(
        model,
        device=device,
        stream_chunk_size=stream_chunk_size,
        left_context_size=left_context_size,
    )
