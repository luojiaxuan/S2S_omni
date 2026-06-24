from __future__ import annotations

import re
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .textgrid import choose_word_tier, nonsilence_intervals, parse_textgrid


SUPPORTED_SRC_LANGS = {"fr", "es", "pt", "de"}
TARGET_LANG = "en"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
LANG_ALIASES = {
    "fra": "fr",
    "fre": "fr",
    "french": "fr",
    "spa": "es",
    "spanish": "es",
    "por": "pt",
    "portuguese": "pt",
    "deu": "de",
    "ger": "de",
    "german": "de",
}


@dataclass(frozen=True)
class HibikiChunk:
    index: int
    source_audio: str
    source_duration_s: float | None = None
    duration_budget_s: float | None = None
    source_text: str = ""
    reference_en_text: str = ""
    compressed_en_text: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> "HibikiChunk":
        source_audio = (
            data.get("source_audio")
            or data.get("source_audio_path")
            or data.get("audio")
            or data.get("audio_path")
        )
        if not source_audio:
            raise ValueError(f"chunk {index} is missing source audio")
        return cls(
            index=int(data.get("chunk_index", data.get("index", index))),
            source_audio=str(source_audio),
            source_duration_s=optional_float(data.get("source_duration_s")),
            duration_budget_s=optional_float(data.get("duration_budget_s")),
            source_text=str(data.get("source_text") or ""),
            reference_en_text=str(data.get("reference_en_text") or data.get("target_text") or ""),
            compressed_en_text=str(data.get("compressed_en_text") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_index": self.index,
            "source_audio": self.source_audio,
            "source_duration_s": self.source_duration_s,
            "duration_budget_s": self.duration_budget_s,
            "source_text": self.source_text,
            "reference_en_text": self.reference_en_text,
            "compressed_en_text": self.compressed_en_text,
        }


@dataclass(frozen=True)
class HibikiSample:
    sample_id: str
    src_lang: str
    source_audio_chunks: list[HibikiChunk]
    source_text: str = ""
    reference_en_text: str = ""
    compressed_en_text: str = ""
    target_en_wav: str = ""
    target_chunk_wavs: list[str] = field(default_factory=list)
    mfa_boundaries: list[dict[str, Any] | None] = field(default_factory=list)
    duration_budget_s: list[float] = field(default_factory=list)
    speech_s2s_rtf: list[float | None] = field(default_factory=list)
    quality_gates: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    base_sample_id: str = ""
    speed_factor: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HibikiSample":
        sample_id = str(data.get("sample_id") or data.get("id") or data.get("utterance_id") or "")
        if not sample_id:
            raise ValueError("sample is missing sample_id")
        src_lang = normalize_src_lang(str(data.get("src_lang") or data.get("source_language") or ""))
        chunks_data = data.get("source_audio_chunks") or data.get("chunks")
        if not isinstance(chunks_data, list) or not chunks_data:
            chunks_data = infer_chunks_from_flat_record(data)
        chunks = [HibikiChunk.from_dict(chunk, idx) for idx, chunk in enumerate(chunks_data)]
        return cls(
            sample_id=safe_id(sample_id),
            src_lang=src_lang,
            source_audio_chunks=chunks,
            source_text=str(data.get("source_text") or ""),
            reference_en_text=str(data.get("reference_en_text") or data.get("target_text") or ""),
            compressed_en_text=str(data.get("compressed_en_text") or ""),
            target_en_wav=str(data.get("target_en_wav") or data.get("target_wav_path") or ""),
            target_chunk_wavs=list(data.get("target_chunk_wavs") or []),
            mfa_boundaries=list(data.get("mfa_boundaries") or []),
            duration_budget_s=[float(x) for x in data.get("duration_budget_s") or []],
            speech_s2s_rtf=[
                optional_float(x) for x in data.get("speech_s2s_rtf") or []
            ],
            quality_gates=dict(data.get("quality_gates") or {}),
            metadata=dict(data.get("metadata") or {}),
            base_sample_id=str(data.get("base_sample_id") or ""),
            speed_factor=optional_float(data.get("speed_factor")),
        )

    def to_dict(self) -> dict[str, Any]:
        out = {
            "sample_id": self.sample_id,
            "src_lang": self.src_lang,
            "source_audio_chunks": [chunk.to_dict() for chunk in self.source_audio_chunks],
            "source_text": self.source_text,
            "reference_en_text": self.reference_en_text,
            "compressed_en_text": self.compressed_en_text,
            "target_en_wav": self.target_en_wav,
            "target_chunk_wavs": self.target_chunk_wavs,
            "mfa_boundaries": self.mfa_boundaries,
            "duration_budget_s": self.duration_budget_s,
            "speech_s2s_rtf": self.speech_s2s_rtf,
            "quality_gates": self.quality_gates,
            "metadata": self.metadata,
        }
        if self.base_sample_id:
            out["base_sample_id"] = self.base_sample_id
        if self.speed_factor is not None:
            out["speed_factor"] = self.speed_factor
        return out

    @property
    def chunk_count(self) -> int:
        return len(self.source_audio_chunks)

    def source_audio_paths(self) -> list[str]:
        return [chunk.source_audio for chunk in self.source_audio_chunks]

    def source_durations_s(self) -> list[float | None]:
        durations: list[float | None] = []
        for chunk in self.source_audio_chunks:
            if chunk.source_duration_s is not None:
                durations.append(chunk.source_duration_s)
                continue
            try:
                durations.append(audio_duration_s(chunk.source_audio))
            except Exception:
                durations.append(None)
        return durations


@dataclass(frozen=True)
class WordBoundary:
    chunk_index: int
    start_s: float
    end_s: float
    word_start: int
    word_end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "start_s": round(float(self.start_s), 6),
            "end_s": round(float(self.end_s), 6),
            "word_start": int(self.word_start),
            "word_end": int(self.word_end),
        }


def normalize_src_lang(lang: str) -> str:
    value = str(lang or "").strip().lower().replace("_", "-")
    value = value.split("-", 1)[0]
    value = LANG_ALIASES.get(value, value)
    if value not in SUPPORTED_SRC_LANGS:
        choices = ", ".join(sorted(SUPPORTED_SRC_LANGS))
        raise ValueError(f"unsupported Hibiki-Zero src_lang {lang!r}; expected one of: {choices}")
    return value


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:180] or "sample"


def audio_duration_s(path: str | Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        with wave.open(str(path), "rb") as handle:
            return float(handle.getnframes()) / float(handle.getframerate())


def infer_chunks_from_flat_record(data: dict[str, Any]) -> list[dict[str, Any]]:
    audio_chunks = (
        data.get("source_audio_chunks")
        or data.get("audio_chunks")
        or data.get("audios")
        or data.get("source_audios")
    )
    if isinstance(audio_chunks, list) and audio_chunks:
        out = []
        for idx, audio in enumerate(audio_chunks):
            if isinstance(audio, dict):
                out.append(audio)
            else:
                out.append({"chunk_index": idx, "source_audio": str(audio)})
        return out
    audio = data.get("source_audio") or data.get("source_audio_path") or data.get("audio")
    if audio:
        return [{"chunk_index": 0, "source_audio": str(audio)}]
    raise ValueError("sample is missing source_audio_chunks or source_audio")


def english_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text or "")]


def english_mfa_transcript(text: str) -> str:
    return " ".join(english_words(text))


def compact_whitespace(text: str) -> str:
    return " ".join(str(text or "").split())


def join_chunk_texts(chunks: Iterable[str]) -> str:
    return compact_whitespace(" ".join(chunk for chunk in chunks if chunk and chunk.strip()))


def chunk_word_spans(chunk_texts: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for text in chunk_texts:
        count = len(english_words(text))
        spans.append((cursor, cursor + count))
        cursor += count
    return spans


def boundaries_from_textgrid(
    textgrid_path: str | Path,
    full_text: str,
    chunk_texts: list[str],
    *,
    min_duration_s: float = 0.08,
    max_word_mismatch_ratio: float = 0.1,
) -> list[WordBoundary | None]:
    expected_words = english_words(full_text)
    if not expected_words:
        raise ValueError("full_text has no English words")
    tiers = parse_textgrid(textgrid_path)
    intervals = nonsilence_intervals(choose_word_tier(tiers))
    interval_words = [english_words(interval.text) for interval in intervals]
    flat_interval_words = [word for words in interval_words for word in words[:1]]
    mismatch = abs(len(flat_interval_words) - len(expected_words)) / max(1, len(expected_words))
    if mismatch > max_word_mismatch_ratio:
        raise ValueError(
            "MFA interval/word mismatch: "
            f"intervals={len(flat_interval_words)} words={len(expected_words)}"
        )
    usable_words = min(len(intervals), len(expected_words))
    out: list[WordBoundary | None] = []
    for idx, (word_start, word_end) in enumerate(chunk_word_spans(chunk_texts)):
        if word_start == word_end:
            out.append(None)
            continue
        if word_start >= usable_words or word_end > usable_words:
            raise ValueError(f"chunk word span out of MFA range: {word_start}:{word_end}/{usable_words}")
        start_s = float(intervals[word_start].start_s)
        end_s = float(intervals[word_end - 1].end_s)
        if end_s <= start_s:
            raise ValueError(f"non-monotonic MFA boundary for chunk {idx}: {start_s} >= {end_s}")
        if end_s - start_s < min_duration_s:
            out.append(None)
            continue
        out.append(
            WordBoundary(
                chunk_index=idx,
                start_s=start_s,
                end_s=end_s,
                word_start=word_start,
                word_end=word_end,
            )
        )
    return out


def duration_budgets_from_source(
    source_durations_s: list[float | None],
    *,
    rtf_threshold: float = 1.0,
    slack_s: float = 0.0,
) -> list[float]:
    budgets = []
    for duration in source_durations_s:
        duration_f = max(0.0, float(duration or 0.0))
        budgets.append(round(duration_f * rtf_threshold + slack_s, 6))
    return budgets


def speech_s2s_rtf_for_chunks(
    target_durations_s: list[float | None],
    source_durations_s: list[float | None],
) -> list[float | None]:
    out: list[float | None] = []
    for target_s, source_s in zip(target_durations_s, source_durations_s, strict=False):
        if target_s is None or source_s is None or source_s <= 0:
            out.append(None)
        else:
            out.append(round(float(target_s) / float(source_s), 6))
    return out


def gate_duration_rtf(
    rtf_values: list[float | None],
    *,
    threshold: float = 1.0,
) -> dict[str, Any]:
    valid = [value for value in rtf_values if value is not None]
    violations = [value for value in valid if value > threshold]
    return {
        "rtf_threshold": threshold,
        "valid_chunks": len(valid),
        "violation_count": len(violations),
        "violation_rate": round(len(violations) / len(valid), 6) if valid else None,
        "max_rtf": round(max(valid), 6) if valid else None,
        "pass": not violations if valid else False,
    }


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
