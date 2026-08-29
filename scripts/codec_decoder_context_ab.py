#!/usr/bin/env python3
"""Decode frozen per-turn MOSS audio codes with reset and continuous codec state."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-path", required=True)
    parser.add_argument("--moss-tts-root", required=True)
    parser.add_argument("--codes-npz", required=True)
    parser.add_argument("--turn-summary-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-frames", type=int, default=3)
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    return parser.parse_args()


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((clipped * 32767.0).astype(np.int16).tobytes())


def load_codes(path: Path) -> tuple[list[str], list[np.ndarray], str]:
    with np.load(path, allow_pickle=False) as archive:
        keys = sorted(archive.files)
        expected = [f"turn_{idx:05d}" for idx in range(len(keys))]
        if keys != expected:
            raise ValueError(f"non-contiguous turn keys: {keys[:3]} ... {keys[-3:]}")
        codes = [archive[key].astype(np.int64, copy=False) for key in keys]
    if not codes:
        raise ValueError("codes archive has no turns")
    channels = codes[0].shape[1] if codes[0].ndim == 2 else None
    digest = hashlib.sha256()
    for key, value in zip(keys, codes, strict=True):
        if value.ndim != 2 or value.shape[1] != channels:
            raise ValueError(f"{key}: expected [frames,{channels}], got {value.shape}")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.astype(np.int16, copy=False).tobytes())
    return keys, codes, digest.hexdigest()


def collect_chunks(decoder) -> list[np.ndarray]:
    chunks = []
    for chunk in decoder.audio_chunks():
        if chunk.numel():
            chunks.append(chunk.detach().float().cpu().numpy().reshape(-1))
    return chunks


def make_decoder(decoder_type, codec, device, chunk_frames):
    return decoder_type(
        codec,
        chunk_frames=chunk_frames,
        overlap_frames=0,
        decode_kwargs={"chunk_duration": -1},
        device=device,
    )


def decode_reset(codec, decoder_type, turn_codes, device, chunk_frames):
    turn_audio = []
    for codes in turn_codes:
        decoder = make_decoder(decoder_type, codec, device, chunk_frames)
        chunks = []
        with codec.streaming(batch_size=1):
            if codes.shape[0]:
                decoder.push_tokens(torch.from_numpy(codes).to(device))
                chunks.extend(collect_chunks(decoder))
            final = decoder.flush()
            if final is not None and final.numel():
                chunks.append(final.detach().float().cpu().numpy().reshape(-1))
        turn_audio.append(np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32))
    return turn_audio


def decode_continuous(codec, decoder_type, turn_codes, device, chunk_frames):
    decoder = make_decoder(decoder_type, codec, device, chunk_frames)
    chunks = []
    with codec.streaming(batch_size=1):
        for codes in turn_codes:
            if codes.shape[0]:
                decoder.push_tokens(torch.from_numpy(codes).to(device))
                chunks.extend(collect_chunks(decoder))
        final = decoder.flush()
        if final is not None and final.numel():
            chunks.append(final.detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def load_turns(path: Path, expected: int) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != 1:
        raise ValueError(f"expected one row summary, got {len(records)}")
    turns = records[0]["turns"]
    if len(turns) != expected:
        raise ValueError(f"summary/code turn mismatch: {len(turns)} != {expected}")
    return turns


def boundary_jump(audio: np.ndarray, at: int, sample_rate: int) -> float:
    width = min(int(0.25 * sample_rate), at, len(audio) - at)
    if width < 256:
        return float("nan")
    left = audio[at - width : at]
    right = audio[at : at + width]
    window = np.hanning(width)
    left_spec = np.log1p(np.abs(np.fft.rfft(left * window)))
    right_spec = np.log1p(np.abs(np.fft.rfft(right * window)))
    denom = np.linalg.norm(left_spec) * np.linalg.norm(right_spec)
    similarity = np.dot(left_spec, right_spec) / denom if denom else 0.0
    return float(1.0 - similarity)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, args.moss_tts_root)
    from mossttsrealtime.streaming_mossttsrealtime import AudioStreamDecoder
    from transformers import AutoModel

    device = torch.device(args.device)
    codec = AutoModel.from_pretrained(args.codec_path, trust_remote_code=True).eval().to(device)
    sample_rate = int(getattr(codec.config, "sampling_rate", 24000))
    samples_per_frame = int(getattr(codec.config, "downsample_rate", sample_rate / 12.5))
    keys, turn_codes, digest = load_codes(Path(args.codes_npz))
    turns = load_turns(Path(args.turn_summary_jsonl), len(turn_codes))

    with torch.inference_mode():
        reset_turn_audio = decode_reset(
            codec, AudioStreamDecoder, turn_codes, device, args.chunk_frames
        )
        continuous_audio = decode_continuous(
            codec, AudioStreamDecoder, turn_codes, device, args.chunk_frames
        )

    reset_audio = np.concatenate(reset_turn_audio)
    reset_lengths = np.asarray([len(audio) for audio in reset_turn_audio], dtype=np.int64)
    expected_lengths = np.asarray(
        [codes.shape[0] * samples_per_frame for codes in turn_codes], dtype=np.int64
    )
    if not np.array_equal(reset_lengths, expected_lengths):
        mismatch = np.flatnonzero(reset_lengths != expected_lengths)[:10].tolist()
        raise RuntimeError(f"reset per-turn length mismatch at turns {mismatch}")
    if len(continuous_audio) != int(expected_lengths.sum()):
        raise RuntimeError(
            f"continuous length {len(continuous_audio)} != expected {expected_lengths.sum()}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_wav(out_dir / "A.wav", reset_audio, sample_rate)
    write_wav(out_dir / "B.wav", continuous_audio, sample_rate)

    a_boundaries = np.cumsum(reset_lengths)[:-1]
    b_boundaries = np.cumsum(expected_lengths)[:-1]
    boundaries = []
    for index, (a_at, b_at) in enumerate(zip(a_boundaries, b_boundaries, strict=True)):
        boundaries.append(
            {
                "boundary_index": index,
                "left_turn": keys[index],
                "right_turn": keys[index + 1],
                "left_text": turns[index].get("text", ""),
                "right_text": turns[index + 1].get("text", ""),
                "a_jump": boundary_jump(reset_audio, int(a_at), sample_rate),
                "b_jump": boundary_jump(continuous_audio, int(b_at), sample_rate),
                "a_sample": int(a_at),
                "b_sample": int(b_at),
            }
        )
    for record in boundaries:
        record["jump_delta_a_minus_b"] = record["a_jump"] - record["b_jump"]

    ranked = sorted(
        boundaries,
        key=lambda item: item["jump_delta_a_minus_b"],
        reverse=True,
    )
    clip_samples = int(args.clip_seconds * sample_rate)
    clips_dir = out_dir / "clips"
    for rank, record in enumerate(ranked[:12], 1):
        for label, audio, at in (
            ("A", reset_audio, record["a_sample"]),
            ("B", continuous_audio, record["b_sample"]),
        ):
            clip = audio[max(0, at - clip_samples) : min(len(audio), at + clip_samples)]
            path = clips_dir / f"{rank:02d}_{record['boundary_index']:05d}_{label}.wav"
            write_wav(path, clip, sample_rate)
            record[f"clip_{label.lower()}"] = str(path)

    report = {
        "audio_code_sha256": digest,
        "num_turns": len(turn_codes),
        "num_boundaries": len(boundaries),
        "sample_rate": sample_rate,
        "samples_per_frame": samples_per_frame,
        "total_code_frames": int(sum(codes.shape[0] for codes in turn_codes)),
        "a_samples": len(reset_audio),
        "b_samples": len(continuous_audio),
        "a_duration_s": len(reset_audio) / sample_rate,
        "b_duration_s": len(continuous_audio) / sample_rate,
        "a_definition": "new AudioStreamDecoder and codec.streaming state for every turn",
        "b_definition": "one AudioStreamDecoder and one codec.streaming state for all turns",
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "boundaries.jsonl").open("w", encoding="utf-8") as handle:
        for record in boundaries:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
