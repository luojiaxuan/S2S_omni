from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TTSBackendSpec:
    name: str
    config_cls: str
    default_model_path: str | None
    relay_backend: str = "shm"

    def to_metadata(self, model_path: str | None = None) -> dict[str, Any]:
        return {
            "tts_backend": self.name,
            "tts_config_cls": self.config_cls,
            "tts_model_path": model_path or self.default_model_path,
            "tts_relay_backend": self.relay_backend,
            "tts_duration_policy": "match_target_text_default_speech_duration",
        }


TTS_BACKENDS: dict[str, TTSBackendSpec] = {
    "qwen3_tts": TTSBackendSpec(
        name="qwen3_tts",
        config_cls="Qwen3TTSPipelineConfig",
        default_model_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ),
    "moss_tts": TTSBackendSpec(
        name="moss_tts",
        config_cls="MossTTSPipelineConfig",
        default_model_path="OpenMOSS-Team/MOSS-TTS-v1.5",
    ),
    "higgs_tts": TTSBackendSpec(
        name="higgs_tts",
        config_cls="GenericTTSPipelineConfig",
        default_model_path=None,
    ),
}


def tts_metadata_for_backend(backend: str, model_path: str | None = None) -> dict[str, Any]:
    try:
        spec = TTS_BACKENDS[backend]
    except KeyError as exc:
        choices = ", ".join(sorted(TTS_BACKENDS))
        raise ValueError(f"unknown TTS backend {backend!r}; expected one of: {choices}") from exc
    metadata = spec.to_metadata(model_path)
    if metadata["tts_model_path"] is None:
        metadata["tts_model_path_required"] = True
    return metadata
