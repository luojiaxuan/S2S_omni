from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AudioSpan:
    path: str
    offset: int | None = None
    frames: int | None = None


def parse_audio_span(uri: str) -> AudioSpan:
    """Parse GigaSpeech-style audio spans such as file.opus:123:456."""
    parts = uri.rsplit(":", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return AudioSpan(path=parts[0], offset=int(parts[1]), frames=int(parts[2]))
    return AudioSpan(path=uri)


def load_audio_span(uri: str, target_sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    span = parse_audio_span(uri)
    audio, sample_rate = _read_with_soundfile(span)
    audio = _to_mono_float32(audio)
    if sample_rate != target_sample_rate:
        audio = resample_audio(audio, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return audio, sample_rate


def resample_audio(audio: np.ndarray, src_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if src_sample_rate == target_sample_rate:
        return audio.astype(np.float32, copy=False)
    if src_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("sample rates must be positive")
    try:
        from scipy.signal import resample_poly

        divisor = math.gcd(src_sample_rate, target_sample_rate)
        up = target_sample_rate // divisor
        down = src_sample_rate // divisor
        return resample_poly(audio, up, down).astype(np.float32)
    except Exception:
        duration = len(audio) / float(src_sample_rate)
        target_len = max(1, int(round(duration * target_sample_rate)))
        source_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        target_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
        return np.interp(target_x, source_x, audio).astype(np.float32)


def _read_with_soundfile(span: AudioSpan) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ModuleNotFoundError as exc:
        raise RuntimeError("soundfile is required to read audio spans") from exc

    path = Path(span.path)
    if not path.exists():
        raise FileNotFoundError(path)
    with sf.SoundFile(path) as handle:
        if span.offset is not None:
            handle.seek(span.offset)
        audio = handle.read(
            frames=-1 if span.frames is None else span.frames,
            dtype="float32",
            always_2d=False,
        )
        sample_rate = int(handle.samplerate)
    return audio, sample_rate


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    if array.ndim != 1:
        array = array.reshape(-1)
    if not array.size:
        raise ValueError("empty audio span")
    peak = float(np.max(np.abs(array)))
    if peak > 1.0:
        array = array / peak
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
