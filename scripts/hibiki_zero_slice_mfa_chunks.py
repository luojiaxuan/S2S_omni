#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.hibiki_zero import (
    HibikiSample,
    WordBoundary,
    boundaries_from_textgrid,
    duration_budgets_from_source,
    gate_duration_rtf,
    speech_s2s_rtf_for_chunks,
)
from s2s_omni.io import read_jsonl
from s2s_omni.rasst import (
    audio_duration_s,
    load_mono_audio,
    sanitize_id,
    speed_audio_ffmpeg,
    write_mono_wav,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slice English TTS target wavs into Hibiki-Zero streaming target chunks."
    )
    parser.add_argument("--tts-manifest", required=True)
    parser.add_argument("--textgrid-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-output", default="sft_sample_manifest.jsonl")
    parser.add_argument("--turn-output", default="sft_turn_manifest.jsonl")
    parser.add_argument("--speed-factors", default="1.0,1.35,1.7,2.0")
    parser.add_argument("--speed-assignment", choices=["cycle", "all"], default="cycle")
    parser.add_argument("--source-sample-rate", type=int, default=16000)
    parser.add_argument("--target-sample-rate", type=int, default=24000)
    parser.add_argument("--rtf-threshold", type=float, default=1.0)
    parser.add_argument("--budget-slack-s", type=float, default=0.0)
    parser.add_argument("--min-target-chunk-s", type=float, default=0.08)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--overwrite-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-existing-source-speed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-edge-silence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def find_textgrid(textgrid_dir: str | Path, sample_id: str) -> Path | None:
    root = Path(textgrid_dir)
    names = [
        f"{sanitize_id(sample_id)}.TextGrid",
        f"{sanitize_id(sample_id)}.textgrid",
        f"{sample_id}.TextGrid",
        f"{sample_id}.textgrid",
    ]
    for name in names:
        path = root / name
        if path.exists():
            return path
    matches = list(root.rglob(f"{sanitize_id(sample_id)}.TextGrid"))
    return matches[0] if matches else None


def chunk_texts(sample: HibikiSample) -> list[str]:
    return [chunk.compressed_en_text for chunk in sample.source_audio_chunks]


def include_edge_silence(
    boundaries: list[WordBoundary | None],
    full_duration_s: float,
) -> list[WordBoundary | None]:
    nonempty = [(idx, boundary) for idx, boundary in enumerate(boundaries) if boundary is not None]
    if not nonempty:
        return boundaries
    out = list(boundaries)
    first_idx, first = nonempty[0]
    last_idx, _ = nonempty[-1]
    out[first_idx] = replace(first, start_s=0.0)
    last = out[last_idx]
    if last is not None:
        out[last_idx] = replace(last, end_s=max(float(last.end_s), full_duration_s))
    return out


def slice_target_wavs(
    sample: HibikiSample,
    boundaries: list[Any],
    output_dir: Path,
    target_sample_rate: int,
) -> tuple[list[str], list[float | None]]:
    wav, sr = load_mono_audio(sample.target_en_wav, target_sample_rate)
    paths: list[str] = []
    durations: list[float | None] = []
    chunk_dir = output_dir / "target_chunk_wav" / sample.sample_id
    for idx, boundary in enumerate(boundaries):
        if boundary is None:
            paths.append("")
            durations.append(None)
            continue
        start = max(0, min(len(wav), int(round(boundary.start_s * sr))))
        end = max(start, min(len(wav), int(round(boundary.end_s * sr))))
        out_path = chunk_dir / f"chunk_{idx:04d}.wav"
        write_mono_wav(out_path, wav[start:end], sr)
        paths.append(str(out_path))
        durations.append(round((end - start) / float(sr), 6))
    return paths, durations


def speed_source_chunks(
    sample: HibikiSample,
    speed: float,
    output_dir: Path,
    sample_rate: int,
    overwrite: bool,
) -> tuple[list[str], list[float | None]]:
    paths: list[str] = []
    durations: list[float | None] = []
    for chunk in sample.source_audio_chunks:
        if speed == 1.0:
            path = chunk.source_audio
        else:
            path_obj = (
                output_dir
                / "source_speed_wav"
                / sample.sample_id
                / f"chunk_{chunk.index:04d}__speed_{speed:g}.wav"
            )
            speed_audio_ffmpeg(
                chunk.source_audio,
                path_obj,
                speed,
                sample_rate=sample_rate,
                overwrite=overwrite,
            )
            path = str(path_obj)
        paths.append(path)
        try:
            durations.append(round(audio_duration_s(path), 6))
        except Exception:
            durations.append(None)
    return paths, durations


def existing_source_chunks(sample: HibikiSample) -> tuple[list[str], list[float | None]]:
    paths: list[str] = []
    durations: list[float | None] = []
    for chunk in sample.source_audio_chunks:
        paths.append(chunk.source_audio)
        if chunk.source_duration_s is not None:
            durations.append(chunk.source_duration_s)
            continue
        try:
            durations.append(round(audio_duration_s(chunk.source_audio), 6))
        except Exception:
            durations.append(None)
    return paths, durations


def build_turn_rows(
    sample_row: dict[str, Any],
    source_paths: list[str],
    source_durations: list[float | None],
    target_paths: list[str],
    target_durations: list[float | None],
) -> list[dict[str, Any]]:
    rows = []
    chunks = sample_row["source_audio_chunks"]
    for idx, chunk in enumerate(chunks):
        rows.append(
            {
                "id": f"{sample_row['sample_id']}__chunk_{idx:04d}",
                "sample_id": sample_row["sample_id"],
                "base_sample_id": sample_row.get("base_sample_id"),
                "speed_factor": sample_row.get("speed_factor"),
                "src_lang": sample_row["src_lang"],
                "tgt_lang": "en",
                "chunk_index": idx,
                "source_audio_chunks": source_paths[: idx + 1],
                "current_source_audio": source_paths[idx],
                "current_source_duration_s": source_durations[idx],
                "target_text": chunk.get("compressed_en_text") or "",
                "target_wav_path": target_paths[idx] if idx < len(target_paths) else "",
                "target_duration_s": target_durations[idx] if idx < len(target_durations) else None,
                "speech_s2s_rtf": sample_row["speech_s2s_rtf"][idx],
                "mfa_boundary": sample_row["mfa_boundaries"][idx],
                "expected_empty_target": not bool(chunk.get("compressed_en_text")),
                "supervision": {
                    "speech_to_speech": bool(target_paths[idx] if idx < len(target_paths) else ""),
                    "text_transcript": bool(chunk.get("compressed_en_text")),
                },
            }
        )
    return rows


def process_sample(
    sample: HibikiSample,
    textgrid_dir: str | Path,
    output_dir: Path,
    args: argparse.Namespace,
    speed: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    textgrid = find_textgrid(textgrid_dir, sample.sample_id)
    if textgrid is None:
        return None, [], {"sample_id": sample.sample_id, "reject_reasons": ["missing_textgrid"]}
    try:
        texts = chunk_texts(sample)
        boundaries = boundaries_from_textgrid(
            textgrid,
            sample.compressed_en_text,
            texts,
            min_duration_s=args.min_target_chunk_s,
        )
        if args.include_edge_silence:
            boundaries = include_edge_silence(boundaries, audio_duration_s(sample.target_en_wav))
        for idx, (text, boundary) in enumerate(zip(texts, boundaries, strict=False)):
            if text and boundary is None:
                raise ValueError(f"nonempty chunk {idx} has no valid MFA boundary")
        target_paths, target_durations = slice_target_wavs(
            sample,
            boundaries,
            output_dir,
            args.target_sample_rate,
        )
        if args.use_existing_source_speed:
            source_paths, source_durations = existing_source_chunks(sample)
        else:
            source_paths, source_durations = speed_source_chunks(
                sample,
                speed,
                output_dir,
                args.source_sample_rate,
                args.overwrite_audio,
            )
        budgets = duration_budgets_from_source(
            source_durations,
            rtf_threshold=args.rtf_threshold,
            slack_s=args.budget_slack_s,
        )
        rtf_values = speech_s2s_rtf_for_chunks(target_durations, source_durations)
        gates = dict(sample.quality_gates)
        gates["duration_gate"] = gate_duration_rtf(rtf_values, threshold=args.rtf_threshold)
        out = sample.to_dict()
        sample_id = sample.sample_id if args.use_existing_source_speed else f"{sample.sample_id}__speed_{speed:g}"
        base_sample_id = sample.base_sample_id or sample.sample_id
        speed_factor = sample.speed_factor if args.use_existing_source_speed else speed
        out.update(
            {
                "sample_id": sample_id,
                "base_sample_id": base_sample_id,
                "speed_factor": speed_factor,
                "source_audio_chunks": [
                    {
                        **chunk.to_dict(),
                        "source_audio": source_paths[idx],
                        "source_duration_s": source_durations[idx],
                        "duration_budget_s": budgets[idx],
                    }
                    for idx, chunk in enumerate(sample.source_audio_chunks)
                ],
                "target_chunk_wavs": target_paths,
                "target_chunk_duration_s": target_durations,
                "mfa_textgrid_path": str(textgrid),
                "mfa_boundaries": [
                    boundary.to_dict() if boundary is not None else None for boundary in boundaries
                ],
                "duration_budget_s": budgets,
                "speech_s2s_rtf": rtf_values,
                "quality_gates": gates,
            }
        )
        return out, build_turn_rows(out, source_paths, source_durations, target_paths, target_durations), None
    except Exception as exc:
        return (
            None,
            [],
            {
                "sample_id": sample.sample_id,
                "speed_factor": sample.speed_factor if args.use_existing_source_speed else speed,
                "reject_reasons": [f"exception:{type(exc).__name__}"],
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    sample_path = output_dir / args.sample_output
    turn_path = output_dir / args.turn_output
    rejected_path = output_dir / "sft_rejected.jsonl"
    for path in [sample_path, turn_path, rejected_path]:
        if path.exists():
            path.unlink()
    speeds = [float(item) for item in args.speed_factors.split(",") if item.strip()]
    if not speeds:
        raise SystemExit("--speed-factors cannot be empty")
    accepted = rejected = source_rows = 0
    for idx, record in enumerate(read_jsonl(args.tts_manifest)):
        if args.max_records and idx >= args.max_records:
            break
        source_rows += 1
        if not record.get("accepted", True):
            continue
        sample = HibikiSample.from_dict(record)
        if args.use_existing_source_speed:
            sample_speeds = [sample.speed_factor if sample.speed_factor is not None else 1.0]
        else:
            sample_speeds = speeds if args.speed_assignment == "all" else [speeds[idx % len(speeds)]]
        for speed in sample_speeds:
            sample_row, turn_rows, reject_row = process_sample(
                sample,
                args.textgrid_dir,
                output_dir,
                args,
                speed,
            )
            if reject_row:
                append_jsonl(rejected_path, reject_row)
                rejected += 1
                continue
            assert sample_row is not None
            append_jsonl(sample_path, sample_row)
            for turn_row in turn_rows:
                append_jsonl(turn_path, turn_row)
            accepted += 1
        if args.log_every > 0 and source_rows % args.log_every == 0:
            print(
                json.dumps(
                    {"source_rows": source_rows, "accepted": accepted, "rejected": rejected},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    summary = {
        "tts_manifest": args.tts_manifest,
        "textgrid_dir": args.textgrid_dir,
        "output_dir": str(output_dir),
        "sample_output": str(sample_path),
        "turn_output": str(turn_path),
        "source_rows": source_rows,
        "accepted": accepted,
        "rejected": rejected,
        "speed_factors": speeds,
        "speed_assignment": args.speed_assignment,
    }
    (output_dir / "slice_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if rejected and not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
