from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import resample_audio


def base_id_from_id(sample_id: str) -> str:
    sample_id = str(sample_id)
    for marker in ("__speed_", "__chunk_"):
        if marker in sample_id:
            return sample_id.split(marker, 1)[0]
    return sample_id


def align_wav_to_num_frames(wav: np.ndarray, num_frames: int, hop_length: int) -> np.ndarray:
    target_len = int(num_frames) * int(hop_length)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.shape[0] > target_len:
        return wav[:target_len]
    if wav.shape[0] < target_len:
        wav = np.pad(wav, (0, target_len - wav.shape[0]))
    return wav.astype(np.float32, copy=False)


def load_mono_wav(path: str | Path, expected_sample_rate: int) -> np.ndarray:
    import soundfile as sf

    wav, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if wav.ndim != 1:
        wav = wav.reshape(-1)
    if sample_rate != expected_sample_rate:
        wav = resample_audio(wav, int(sample_rate), int(expected_sample_rate))
    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    return np.clip(wav, -1.0, 1.0).astype(np.float32)


def resolve_manifest_path(raw_path: str | Path, manifest_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return Path(manifest_path).resolve().parent / path
