#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed-size wav chunks from GigaSpeech TSV spans.")
    parser.add_argument("--tsv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=260622)
    parser.add_argument("--chunk-s", type=float, default=1.92)
    parser.add_argument("--max-chunks-per-sample", type=int, default=6)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--span-unit-rate", type=float, default=16000.0)
    parser.add_argument("--min-duration-s", type=float, default=5.0)
    parser.add_argument("--max-scan-rows", type=int, default=200000)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def parse_audio_span(uri: str) -> tuple[str, int | None, int | None]:
    parts = uri.rsplit(":", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0], int(parts[1]), int(parts[2])
    return uri, None, None


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180] or "sample"


def read_candidates(args: argparse.Namespace) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    with Path(args.tsv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for index, row in enumerate(reader):
            if args.max_scan_rows > 0 and index >= args.max_scan_rows:
                break
            if row.get("src_lang") != "en" or row.get("tgt_lang") != "zh":
                continue
            if not row.get("src_text", "").strip() or not row.get("tgt_text", "").strip():
                continue
            audio_path, offset, frames = parse_audio_span(row.get("audio", ""))
            if not audio_path or offset is None or frames is None:
                continue
            duration_s = frames / float(args.span_unit_rate)
            if duration_s < args.min_duration_s:
                continue
            if not Path(audio_path).exists():
                continue
            candidates.append(row)
    return candidates


def run_ffmpeg(
    source: str,
    start_s: float,
    duration_s: float,
    sample_rate: int,
    output: Path,
    overwrite: bool,
) -> None:
    if output.exists() and not overwrite:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.6f}",
        "-i",
        source,
        "-t",
        f"{duration_s:.6f}",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output),
    ]
    subprocess.run(cmd, check=True)


def build_stream(row: dict[str, str], args: argparse.Namespace, audio_dir: Path) -> dict[str, Any]:
    audio_path, offset, frames = parse_audio_span(row["audio"])
    assert offset is not None and frames is not None
    span_start_s = offset / float(args.span_unit_rate)
    span_duration_s = frames / float(args.span_unit_rate)
    chunk_count = min(args.max_chunks_per_sample, max(1, int(span_duration_s // args.chunk_s)))
    chunks: list[dict[str, Any]] = []
    stream_dir = audio_dir / safe_name(row["id"])
    for chunk_index in range(chunk_count):
        rel_start_s = chunk_index * args.chunk_s
        duration_s = min(args.chunk_s, span_duration_s - rel_start_s)
        if duration_s <= 0:
            break
        chunk_id = f"{row['id']}__chunk_{chunk_index:04d}"
        wav_path = stream_dir / f"{safe_name(chunk_id)}.wav"
        run_ffmpeg(
            audio_path,
            span_start_s + rel_start_s,
            duration_s,
            args.sample_rate,
            wav_path,
            args.overwrite,
        )
        chunks.append(
            {
                "id": chunk_id,
                "audio_path": str(wav_path),
                "start_s": round(rel_start_s, 6),
                "end_s": round(rel_start_s + duration_s, 6),
                "duration_s": round(duration_s, 6),
            }
        )
    return {
        "id": row["id"],
        "audio_span": row["audio"],
        "source_text": row.get("src_text", ""),
        "reference_translation": row.get("tgt_text", ""),
        "src_lang": row.get("src_lang", "en"),
        "tgt_lang": row.get("tgt_lang", "zh"),
        "span_duration_s": round(span_duration_s, 6),
        "chunk_s": args.chunk_s,
        "sample_rate": args.sample_rate,
        "chunks": chunks,
    }


def main() -> None:
    args = parse_args()
    candidates = read_candidates(args)
    if not candidates:
        raise RuntimeError("no GigaSpeech candidates found")
    rng = random.Random(args.seed)
    rows = rng.sample(candidates, min(args.num_samples, len(candidates)))
    audio_dir = Path(args.audio_dir)
    streams = [build_stream(row, args, audio_dir) for row in rows]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for stream in streams:
            handle.write(json.dumps(stream, ensure_ascii=False) + "\n")
    print(json.dumps({"streams": len(streams), "chunks": sum(len(s["chunks"]) for s in streams)}, indent=2))


if __name__ == "__main__":
    main()
