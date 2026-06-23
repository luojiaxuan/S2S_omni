#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.rasst import (
    DEFAULT_RASST_DEV,
    DEFAULT_RASST_TRAIN,
    audio_duration_s,
    frame_slice_for_time_span,
    iter_rasst_rows,
    sanitize_id,
    speed_audio_ffmpeg,
    write_mono_wav,
)
from s2s_omni.textgrid import chunk_time_spans_from_textgrid


S2ST_SOFTWAV_SYSTEM = (
    "You are a professional simultaneous speech-to-speech interpreter. "
    "For each incoming English audio chunk, produce natural spoken Mandarin audio. "
    "Keep the Mandarin transcript concise, modern, and aligned to the current streaming chunk."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build turn-level soft-wav SFT manifest from plain RASST baseline data."
    )
    parser.add_argument("--rasst-jsonl", default=DEFAULT_RASST_TRAIN)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--textgrid-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", default="turn_manifest.jsonl")
    parser.add_argument("--speed-factors", default="1.0,1.35,1.7,2.0")
    parser.add_argument(
        "--speed-assignment",
        choices=["cycle", "all"],
        default="cycle",
        help="cycle keeps one speed variant per source row; all emits all speed variants.",
    )
    parser.add_argument("--source-sample-rate", type=int, default=16000)
    parser.add_argument("--target-sample-rate", type=int, default=24000)
    parser.add_argument("--codec-frame-rate", type=float, default=12.5)
    parser.add_argument("--hop-length", type=int, default=1920)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-turns-per-row", type=int, default=0)
    parser.add_argument("--overwrite-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-missing-targets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--system-prompt",
        default=S2ST_SOFTWAV_SYSTEM,
        help="System prompt written into S2ST soft-wav training messages.",
    )
    parser.add_argument(
        "--preserve-source-system-prompt",
        action="store_true",
        help="Use the original RASST system prompt instead of the S2ST soft-wav prompt.",
    )
    parser.add_argument("--dev", action="store_true", help=f"Shortcut for --rasst-jsonl {DEFAULT_RASST_DEV}")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def load_target_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("accepted") and row.get("row_id"):
                out[str(row["row_id"])] = row
    return out


def find_textgrid(textgrid_dir: str | Path, row_id: str) -> Path | None:
    root = Path(textgrid_dir)
    names = [
        f"{sanitize_id(row_id)}.TextGrid",
        f"{sanitize_id(row_id)}.textgrid",
        f"{row_id}.TextGrid",
        f"{row_id}.textgrid",
    ]
    for name in names:
        path = root / name
        if path.exists():
            return path
    matches = list(root.rglob(f"{sanitize_id(row_id)}.TextGrid"))
    if matches:
        return matches[0]
    return None


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def load_wav(path: str | Path, sample_rate: int) -> np.ndarray:
    from s2s_omni.rasst import load_mono_audio

    wav, _ = load_mono_audio(path, target_sample_rate=sample_rate)
    return wav


def slice_target_assets(
    *,
    target_row: dict[str, Any],
    time_span: Any,
    chunk_dir: Path,
    row_id: str,
    chunk_index: int,
    target_sample_rate: int,
    codec_frame_rate: float,
) -> tuple[str, str, float, int, int]:
    target_wav = load_wav(target_row["target_wav_path"], target_sample_rate)
    codes_path_value = target_row.get("target_codes_path")
    codes = None
    total_frames = 0
    if codes_path_value:
        codes = np.load(codes_path_value)
        if codes.ndim == 3 and codes.shape[0] == 1:
            codes = codes[0]
        total_frames = int(codes.shape[1])

    sample_start = max(0, min(len(target_wav), int(round(time_span.start_s * target_sample_rate))))
    sample_end = max(sample_start, min(len(target_wav), int(round(time_span.end_s * target_sample_rate))))
    frame_start = 0
    frame_end = 0
    if codes is not None:
        frame_start, frame_end = frame_slice_for_time_span(
            time_span.start_s,
            time_span.end_s,
            codec_frame_rate,
            total_frames,
        )
    safe = sanitize_id(row_id)
    wav_path = chunk_dir / "target_wav" / safe / f"chunk_{chunk_index:04d}.wav"
    codes_path = chunk_dir / "target_codes" / safe / f"chunk_{chunk_index:04d}.npy"
    write_mono_wav(wav_path, target_wav[sample_start:sample_end], target_sample_rate)
    if codes is not None:
        codes_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(codes_path, codes[:, frame_start:frame_end].astype(np.int16, copy=False))
    duration_s = (sample_end - sample_start) / float(target_sample_rate)
    return (
        str(wav_path),
        str(codes_path) if codes is not None else "",
        round(duration_s, 6),
        frame_start,
        frame_end,
    )


def source_speed_path(output_dir: Path, row_id: str, chunk_index: int, speed: float) -> Path:
    return (
        output_dir
        / "source_speed_wav"
        / sanitize_id(row_id)
        / f"chunk_{chunk_index:04d}__speed_{speed:g}.wav"
    )


def messages_for_turn(row: Any, upto_turn: int, system_prompt: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    for turn in row.turns[: upto_turn + 1]:
        messages.append({"role": "user", "content": "<audio>"})
        messages.append({"role": "assistant", "content": turn.assistant_text})
    return messages


def main() -> None:
    args = parse_args()
    if args.dev:
        args.rasst_jsonl = DEFAULT_RASST_DEV
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output
    rejected_path = output_dir / "turn_manifest_rejected.jsonl"
    if output_path.exists():
        output_path.unlink()
    if rejected_path.exists():
        rejected_path.unlink()

    speed_factors = [float(item) for item in args.speed_factors.split(",") if item.strip()]
    if not speed_factors:
        raise SystemExit("--speed-factors cannot be empty")
    target_rows = load_target_manifest(args.target_manifest)
    emitted = 0
    rejected = 0
    source_rows = 0

    for row_index, row in enumerate(iter_rasst_rows(args.rasst_jsonl, args.max_records)):
        source_rows += 1
        target_row = target_rows.get(row.row_id)
        if target_row is None:
            rejected += 1
            append_jsonl(
                rejected_path,
                {"row_id": row.row_id, "reject_reasons": ["missing_target_manifest"]},
            )
            if not args.allow_missing_targets:
                continue
        textgrid_path = find_textgrid(args.textgrid_dir, row.row_id)
        if target_row is None or textgrid_path is None:
            rejected += int(textgrid_path is None)
            append_jsonl(
                rejected_path,
                {"row_id": row.row_id, "reject_reasons": ["missing_textgrid"]},
            )
            continue
        try:
            time_spans = chunk_time_spans_from_textgrid(
                textgrid_path,
                row.full_target_text,
                row.target_char_spans,
            )
        except Exception as exc:
            rejected += 1
            append_jsonl(
                rejected_path,
                {
                    "row_id": row.row_id,
                    "textgrid_path": str(textgrid_path),
                    "reject_reasons": [f"textgrid_exception:{type(exc).__name__}"],
                    "error": str(exc),
                },
            )
            continue

        row_speeds = (
            speed_factors
            if args.speed_assignment == "all"
            else [speed_factors[row_index % len(speed_factors)]]
        )
        for speed in row_speeds:
            speed_audio_paths: list[str] = []
            for turn in row.turns:
                speed_path = source_speed_path(output_dir, row.row_id, turn.index, speed)
                speed_audio_ffmpeg(
                    turn.audio_path,
                    speed_path,
                    speed,
                    sample_rate=args.source_sample_rate,
                    overwrite=args.overwrite_audio,
                )
                speed_audio_paths.append(str(speed_path))

            for turn in row.turns:
                if args.max_turns_per_row > 0 and turn.index >= args.max_turns_per_row:
                    break
                span = time_spans[turn.index]
                system_prompt = row.system if args.preserve_source_system_prompt else args.system_prompt
                record: dict[str, Any] = {
                    "id": f"{sanitize_id(row.row_id)}__speed_{speed:g}__chunk_{turn.index:04d}",
                    "task_type": "s2st_softwav",
                    "row_id": row.row_id,
                    "chunk_index": turn.index,
                    "speed_factor": speed,
                    "messages": messages_for_turn(row, turn.index, system_prompt),
                    "audios": speed_audio_paths[: turn.index + 1],
                    "current_source_audio": speed_audio_paths[turn.index],
                    "current_source_duration_s": round(audio_duration_s(speed_audio_paths[turn.index]), 6),
                    "target_text": turn.assistant_text,
                    "assistant_text_target": turn.assistant_text,
                    "assistant_text_role": "transcript_for_speech_generation",
                    "assistant_output_modality": "speech",
                    "target_char_span": list(row.target_char_spans[turn.index]),
                    "target_full_text": row.full_target_text,
                    "target_full_wav_path": target_row.get("target_wav_path"),
                    "target_full_codes_path": target_row.get("target_codes_path"),
                    "mfa_textgrid_path": str(textgrid_path),
                    "source_path": row.source_path,
                    "expected_empty_target": not bool(turn.assistant_text),
                    "codec_frame_rate": args.codec_frame_rate,
                    "target_sample_rate": args.target_sample_rate,
                    "source_sample_rate": args.source_sample_rate,
                    "hop_length": args.hop_length,
                    "supervision": {
                        "text_ce": True,
                        "codec_ce": bool(target_row.get("target_codes_path")),
                        "soft_code2wav_wav_loss": bool(turn.assistant_text),
                        "eos_loss": not bool(turn.assistant_text),
                        "speech_target_field": "target_wav_path" if turn.assistant_text else "",
                    },
                }
                if span is not None and turn.assistant_text:
                    wav_path, codes_path, duration_s, frame_start, frame_end = slice_target_assets(
                        target_row=target_row,
                        time_span=span,
                        chunk_dir=output_dir,
                        row_id=row.row_id,
                        chunk_index=turn.index,
                        target_sample_rate=args.target_sample_rate,
                        codec_frame_rate=args.codec_frame_rate,
                    )
                    update = {
                        "target_wav_path": wav_path,
                        "target_audio_path": wav_path,
                        "target_duration_s": duration_s,
                        "target_audio_duration_s": duration_s,
                        "target_time_span": {
                            "start_s": round(span.start_s, 6),
                            "end_s": round(span.end_s, 6),
                            "unit_start": span.unit_start,
                            "unit_end": span.unit_end,
                        },
                        "speech_s2s_rtf_target": (
                            round(duration_s / max(1.0e-6, record["current_source_duration_s"]), 6)
                        ),
                    }
                    if codes_path:
                        update.update(
                            {
                                "target_codes_path": codes_path,
                                "target_code_frame_start": frame_start,
                                "target_code_frame_end": frame_end,
                            }
                        )
                    record.update(update)
                    record["supervision"]["codec_ce"] = bool(codes_path)
                append_jsonl(output_path, record)
                emitted += 1
        if args.log_every > 0 and source_rows % args.log_every == 0:
            print(
                json.dumps(
                    {"source_rows": source_rows, "emitted_turns": emitted, "rejected": rejected},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = {
        "rasst_jsonl": args.rasst_jsonl,
        "target_manifest": args.target_manifest,
        "textgrid_dir": args.textgrid_dir,
        "output": str(output_path),
        "source_rows": source_rows,
        "emitted_turns": emitted,
        "rejected": rejected,
        "speed_factors": speed_factors,
        "speed_assignment": args.speed_assignment,
        "system_prompt": args.system_prompt,
        "preserve_source_system_prompt": args.preserve_source_system_prompt,
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
