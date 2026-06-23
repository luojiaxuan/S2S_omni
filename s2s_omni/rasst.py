from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .audio import resample_audio


AUDIO_MARKER = "<audio>"
DEFAULT_RASST_TRAIN = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline.jsonl"
DEFAULT_RASST_DEV = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline_dev.jsonl"


@dataclass(frozen=True)
class RasstTurn:
    index: int
    user_content: str
    assistant_text: str
    audio_path: str


@dataclass(frozen=True)
class RasstRow:
    row_id: str
    source_path: str
    index: int
    raw: dict[str, Any]
    system: str
    turns: list[RasstTurn]
    full_target_text: str
    target_char_spans: list[tuple[int, int]]


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def stable_row_id(row: dict[str, Any], index: int) -> str:
    explicit = row.get("utter_id") or row.get("id") or row.get("row_id")
    if explicit:
        return sanitize_id(str(explicit))
    audios = row.get("audios") or []
    if audios:
        inferred = infer_utter_id_from_audio(str(audios[0]))
        if inferred:
            return sanitize_id(inferred)
    return f"row_{index:06d}"


def infer_utter_id_from_audio(audio_path: str) -> str | None:
    parts = Path(audio_path).parts
    if len(parts) < 3:
        return None
    try:
        audio_id = parts[-3]
        window = parts[-2]
        if not audio_id or not window:
            return None
        return f"{audio_id}_{window}"
    except IndexError:
        return None


def sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180] or "sample"


def normalize_target_text(text: str) -> str:
    # Plain baseline data has no XML-style term tags. Keep content as-is and only
    # collapse whitespace that is not meaningful for Mandarin speech.
    return "".join(str(text or "").split())


def target_chunks_from_messages(
    messages: list[dict[str, Any]], audios: list[str]
) -> tuple[str, list[RasstTurn], list[tuple[int, int]]]:
    turns: list[RasstTurn] = []
    audio_index = 0
    pending_user = ""
    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if role == "system":
            continue
        if role == "user":
            pending_user = str(content)
            continue
        if role != "assistant":
            continue
        if audio_index >= len(audios):
            raise ValueError("assistant turn count exceeds audios length")
        turns.append(
            RasstTurn(
                index=len(turns),
                user_content=pending_user,
                assistant_text=normalize_target_text(str(content)),
                audio_path=str(audios[audio_index]),
            )
        )
        audio_index += 1
        pending_user = ""
    if audio_index != len(audios):
        raise ValueError(f"audios length {len(audios)} does not match assistant turns {audio_index}")

    cursor = 0
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    for turn in turns:
        start = cursor
        if turn.assistant_text:
            parts.append(turn.assistant_text)
            cursor += len(turn.assistant_text)
        spans.append((start, cursor))
    return "".join(parts), turns, spans


def parse_rasst_row(record: dict[str, Any], source_path: str | Path, index: int) -> RasstRow:
    messages = record.get("messages")
    audios = record.get("audios")
    if not isinstance(messages, list) or not isinstance(audios, list):
        raise ValueError("RASST row must contain list fields: messages and audios")
    system = ""
    if messages and messages[0].get("role") == "system":
        system = str(messages[0].get("content") or "")
    full_target, turns, spans = target_chunks_from_messages(messages, [str(a) for a in audios])
    return RasstRow(
        row_id=stable_row_id(record, index),
        source_path=str(source_path),
        index=index,
        raw=record,
        system=system,
        turns=turns,
        full_target_text=full_target,
        target_char_spans=spans,
    )


def iter_rasst_rows(path: str | Path, max_records: int = 0) -> Iterable[RasstRow]:
    count = 0
    for index, record in enumerate(read_jsonl(path)):
        row = parse_rasst_row(record, path, index)
        yield row
        count += 1
        if max_records > 0 and count >= max_records:
            break


def make_qwen_audio_messages(row: RasstRow, upto_turn: int | None = None) -> list[dict[str, Any]]:
    if upto_turn is None:
        upto_turn = len(row.turns) - 1
    messages: list[dict[str, Any]] = [{"role": "system", "content": row.system}]
    for turn in row.turns[: upto_turn + 1]:
        user_text = str(turn.user_content).replace(AUDIO_MARKER, "").strip()
        content: list[dict[str, str]] = [{"type": "audio", "audio": turn.audio_path}]
        if user_text:
            content.append({"type": "text", "text": user_text})
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": turn.assistant_text})
    return messages


def make_qwen_tts_messages(target_text: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": "You are a Mandarin text-to-speech engine. Speak the assistant text naturally.",
        },
        {"role": "user", "content": "Please speak the following Chinese text."},
        {"role": "assistant", "content": target_text},
    ]


def audio_duration_s(path: str | Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def load_mono_audio(path: str | Path, target_sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    import soundfile as sf

    wav, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if wav.ndim != 1:
        wav = wav.reshape(-1)
    if target_sample_rate and int(sample_rate) != int(target_sample_rate):
        wav = resample_audio(wav, int(sample_rate), int(target_sample_rate))
        sample_rate = int(target_sample_rate)
    return np.nan_to_num(wav).astype(np.float32), int(sample_rate)


def write_mono_wav(path: str | Path, wav: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(wav, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    sf.write(str(path), np.clip(audio, -1.0, 1.0), int(sample_rate), subtype="PCM_16")


def speed_audio_ffmpeg(
    input_path: str | Path,
    output_path: str | Path,
    speed_factor: float,
    sample_rate: int = 16000,
    overwrite: bool = False,
) -> None:
    if speed_factor <= 0:
        raise ValueError("speed_factor must be positive")
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filters = atempo_filters(speed_factor)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-filter:a",
        filters,
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def atempo_filters(speed_factor: float) -> str:
    # ffmpeg atempo accepts [0.5, 100] in modern builds, but chaining in [0.5, 2]
    # is safer across older installs.
    value = float(speed_factor)
    pieces: list[float] = []
    while value > 2.0:
        pieces.append(2.0)
        value /= 2.0
    while value < 0.5:
        pieces.append(0.5)
        value /= 0.5
    pieces.append(value)
    return ",".join(f"atempo={piece:.8g}" for piece in pieces)


def frame_slice_for_time_span(
    start_s: float,
    end_s: float,
    frame_rate: float,
    total_frames: int,
) -> tuple[int, int]:
    start = max(0, min(total_frames, int(math.floor(start_s * frame_rate))))
    end = max(start, min(total_frames, int(math.ceil(end_s * frame_rate))))
    return start, end

