from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Timing:
    src_start_s: float = 0.0
    src_end_s: float | None = None
    tgt_start_s: float | None = None
    tgt_end_s: float | None = None

    @property
    def src_duration_s(self) -> float | None:
        if self.src_end_s is None:
            return None
        return max(0.0, self.src_end_s - self.src_start_s)

    @property
    def tgt_duration_s(self) -> float | None:
        if self.tgt_start_s is None or self.tgt_end_s is None:
            return None
        return max(0.0, self.tgt_end_s - self.tgt_start_s)

    def scaled_source(self, speed_factor: float) -> "Timing":
        if speed_factor <= 0:
            raise ValueError("speed_factor must be positive")
        if self.src_end_s is None:
            return Timing(
                src_start_s=self.src_start_s,
                src_end_s=None,
                tgt_start_s=self.tgt_start_s,
                tgt_end_s=self.tgt_end_s,
            )
        duration = self.src_duration_s or 0.0
        return Timing(
            src_start_s=self.src_start_s,
            src_end_s=self.src_start_s + duration / speed_factor,
            tgt_start_s=self.tgt_start_s,
            tgt_end_s=self.tgt_end_s,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Timing":
        data = data or {}
        return cls(
            src_start_s=float(data.get("src_start_s", 0.0)),
            src_end_s=_optional_float(data.get("src_end_s")),
            tgt_start_s=_optional_float(data.get("tgt_start_s")),
            tgt_end_s=_optional_float(data.get("tgt_end_s")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "src_start_s": self.src_start_s,
            "src_end_s": self.src_end_s,
            "tgt_start_s": self.tgt_start_s,
            "tgt_end_s": self.tgt_end_s,
        }


@dataclass
class CompressionTarget:
    mode: str = "concise"
    target_ratio: float | None = None
    max_target_chars: int | None = None
    max_target_words: int | None = None
    max_target_duration_s: float | None = None
    max_target_wpm: float | None = 185.0
    max_end_lag_s: float | None = 1.5
    preserve: list[str] = field(default_factory=list)
    compressible: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CompressionTarget":
        data = data or {}
        return cls(
            mode=str(data.get("mode", "concise")),
            target_ratio=_optional_float(data.get("target_ratio")),
            max_target_chars=_optional_int(data.get("max_target_chars")),
            max_target_words=_optional_int(data.get("max_target_words")),
            max_target_duration_s=_optional_float(data.get("max_target_duration_s")),
            max_target_wpm=_optional_float(data.get("max_target_wpm", 185.0)),
            max_end_lag_s=_optional_float(data.get("max_end_lag_s", 1.5)),
            preserve=list(data.get("preserve", [])),
            compressible=list(data.get("compressible", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "target_ratio": self.target_ratio,
            "max_target_chars": self.max_target_chars,
            "max_target_words": self.max_target_words,
            "max_target_duration_s": self.max_target_duration_s,
            "max_target_wpm": self.max_target_wpm,
            "max_end_lag_s": self.max_end_lag_s,
            "preserve": self.preserve,
            "compressible": self.compressible,
        }


@dataclass
class S2SSample:
    id: str
    src_lang: str
    tgt_lang: str
    source_text: str
    reference_translation: str | None = None
    compressed_translation: str | None = None
    candidate_translation: str | None = None
    core_meanings: list[str] = field(default_factory=list)
    must_keep_terms: list[str] = field(default_factory=list)
    omitted_noncritical: list[str] = field(default_factory=list)
    audio_path: str | None = None
    target: CompressionTarget = field(default_factory=CompressionTarget)
    timing: Timing = field(default_factory=Timing)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "S2SSample":
        sample_id = str(data.get("id") or data.get("sample_id") or "")
        if not sample_id:
            raise ValueError("sample is missing id")
        source_text = str(data.get("source_text") or "")
        if not source_text:
            raise ValueError(f"sample {sample_id} is missing source_text")
        return cls(
            id=sample_id,
            src_lang=str(data.get("src_lang", "auto")),
            tgt_lang=str(data.get("tgt_lang", "en")),
            source_text=source_text,
            reference_translation=_optional_str(data.get("reference_translation")),
            compressed_translation=_optional_str(data.get("compressed_translation")),
            candidate_translation=_optional_str(
                data.get("candidate_translation") or data.get("prediction")
            ),
            core_meanings=list(data.get("core_meanings", [])),
            must_keep_terms=list(data.get("must_keep_terms", [])),
            omitted_noncritical=list(data.get("omitted_noncritical", [])),
            audio_path=_optional_str(data.get("audio_path")),
            target=CompressionTarget.from_dict(data.get("target") or data.get("compression_target")),
            timing=Timing.from_dict(data.get("timing")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
            "source_text": self.source_text,
            "reference_translation": self.reference_translation,
            "compressed_translation": self.compressed_translation,
            "candidate_translation": self.candidate_translation,
            "core_meanings": self.core_meanings,
            "must_keep_terms": self.must_keep_terms,
            "omitted_noncritical": self.omitted_noncritical,
            "audio_path": self.audio_path,
            "target": self.target.to_dict(),
            "timing": self.timing.to_dict(),
            "metadata": self.metadata,
        }

    def preferred_target_text(self) -> str:
        return (
            self.compressed_translation
            or self.candidate_translation
            or self.reference_translation
            or ""
        )


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value)
    return value if value else None
