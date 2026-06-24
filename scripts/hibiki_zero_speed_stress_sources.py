#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.hibiki_zero import HibikiSample, duration_budgets_from_source
from s2s_omni.io import read_jsonl
from s2s_omni.rasst import audio_duration_s, sanitize_id, speed_audio_ffmpeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand Hibiki-Zero source manifests with source speed-stress budgets."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rejected-output", default="")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--speed-factors", default="1.0,1.35,1.7,2.0")
    parser.add_argument("--speed-assignment", choices=["cycle", "all"], default="cycle")
    parser.add_argument("--source-sample-rate", type=int, default=16000)
    parser.add_argument("--rtf-threshold", type=float, default=1.0)
    parser.add_argument("--budget-slack-s", type=float, default=0.0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--overwrite-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
        handle.flush()


def speed_source_chunks(
    sample: HibikiSample,
    speed: float,
    audio_dir: Path,
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
                audio_dir
                / sanitize_id(sample.sample_id)
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


def build_speed_row(
    sample: HibikiSample,
    speed: float,
    audio_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_paths, source_durations = speed_source_chunks(
        sample,
        speed,
        audio_dir,
        args.source_sample_rate,
        args.overwrite_audio,
    )
    budgets = duration_budgets_from_source(
        source_durations,
        rtf_threshold=args.rtf_threshold,
        slack_s=args.budget_slack_s,
    )
    metadata = dict(sample.metadata)
    metadata.update({"speed_factor": speed, "base_sample_id": sample.sample_id})
    out = sample.to_dict()
    out.update(
        {
            "sample_id": f"{sample.sample_id}__speed_{speed:g}",
            "base_sample_id": sample.sample_id,
            "speed_factor": speed,
            "source_audio_chunks": [
                {
                    **chunk.to_dict(),
                    "source_audio": source_paths[idx],
                    "source_duration_s": source_durations[idx],
                    "duration_budget_s": budgets[idx],
                }
                for idx, chunk in enumerate(sample.source_audio_chunks)
            ],
            "duration_budget_s": budgets,
            "metadata": metadata,
        }
    )
    return out


def main() -> None:
    args = parse_args()
    speeds = [float(item) for item in args.speed_factors.split(",") if item.strip()]
    if not speeds:
        raise SystemExit("--speed-factors cannot be empty")
    output = Path(args.output)
    rejected = Path(args.rejected_output) if args.rejected_output else output.with_suffix(".rejected.jsonl")
    audio_dir = Path(args.audio_dir)
    for path in [output, rejected]:
        if path.exists():
            path.unlink()

    accepted = 0
    rejected_count = 0
    source_rows = 0
    for idx, record in enumerate(read_jsonl(args.input)):
        if args.max_records and idx >= args.max_records:
            break
        source_rows += 1
        try:
            sample = HibikiSample.from_dict(record)
            sample_speeds = speeds if args.speed_assignment == "all" else [speeds[idx % len(speeds)]]
            for speed in sample_speeds:
                append_jsonl(output, build_speed_row(sample, speed, audio_dir, args))
                accepted += 1
        except Exception as exc:
            append_jsonl(
                rejected,
                {
                    "sample_id": record.get("sample_id") or record.get("id"),
                    "reject_reasons": [f"exception:{type(exc).__name__}"],
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            rejected_count += 1
        if args.log_every > 0 and source_rows % args.log_every == 0:
            print(
                json.dumps(
                    {"source_rows": source_rows, "accepted": accepted, "rejected": rejected_count},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    summary = {
        "input": args.input,
        "output": str(output),
        "rejected_output": str(rejected),
        "audio_dir": str(audio_dir),
        "source_rows": source_rows,
        "accepted": accepted,
        "rejected": rejected_count,
        "speed_factors": speeds,
        "speed_assignment": args.speed_assignment,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if rejected_count and not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
