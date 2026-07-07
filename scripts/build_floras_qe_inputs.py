#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.floras_qe import read_jsonl, row_chunk_ms, row_key, row_model, write_jsonl


BOUNDARY_RE = re.compile(r"[\s,.;:!?，。！？；：、]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reference-free QE inputs from FLORAS full compare rows.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--compare-metrics", required=True)
    parser.add_argument("--output-segments", required=True)
    parser.add_argument("--output-runs", required=True)
    parser.add_argument("--max-source-chars", type=int, default=220)
    parser.add_argument("--max-hypothesis-chars", type=int, default=160)
    return parser.parse_args()


def clean_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def split_text(text: str, parts: int) -> list[str]:
    text = clean_text(text)
    if parts <= 1 or not text:
        return [text]
    chunks: list[str] = []
    start = 0
    length = len(text)
    for idx in range(parts - 1):
        target = round(length * (idx + 1) / parts)
        window_start = max(start + 1, target - 120)
        window_end = min(length - 1, target + 120)
        cut = target
        best_distance = length
        for match in BOUNDARY_RE.finditer(text, window_start, window_end):
            distance = abs(match.end() - target)
            if distance < best_distance:
                best_distance = distance
                cut = match.end()
        chunks.append(text[start:cut].strip())
        start = cut
    chunks.append(text[start:].strip())
    return chunks


def segment_count(source_text: str, hypothesis_text: str, max_source_chars: int, max_hypothesis_chars: int) -> int:
    source_parts = math.ceil(max(1, len(source_text)) / max_source_chars)
    hypothesis_parts = math.ceil(max(1, len(hypothesis_text)) / max_hypothesis_chars)
    return max(1, source_parts, hypothesis_parts)


def main() -> None:
    args = parse_args()
    manifest_by_run: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(args.manifest):
        if row.get("run_id") is None:
            continue
        run_id = str(row["run_id"])
        if run_id in manifest_by_run:
            raise SystemExit(f"duplicate run_id in manifest: {run_id}")
        manifest_by_run[run_id] = row
    seen: set[str] = set()
    run_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    for row in read_jsonl(args.compare_metrics):
        key = row_key(row)
        if key in seen:
            continue
        seen.add(key)
        run_id = str(row.get("run_id") or "")
        manifest = manifest_by_run.get(run_id, {})
        source_text = clean_text(manifest.get("source_transcript"))
        hypothesis_text = clean_text(row.get("candidate_text") or row.get("hypothesis_text"))
        if not source_text or not hypothesis_text:
            raise SystemExit(f"missing source or hypothesis for {key}")
        parts = segment_count(source_text, hypothesis_text, args.max_source_chars, args.max_hypothesis_chars)
        source_chunks = split_text(source_text, parts)
        hypothesis_chunks = split_text(hypothesis_text, parts)
        if len(source_chunks) != parts or len(hypothesis_chunks) != parts:
            raise SystemExit(f"bad segmentation for {key}: {len(source_chunks)} vs {len(hypothesis_chunks)}")
        run_rows.append(
            {
                "qe_row_key": key,
                "run_id": run_id,
                "model": row_model(row),
                "chunk_ms": row_chunk_ms(row),
                "eval_label": row.get("eval_label"),
                "speed_factor": row.get("speed_factor"),
                "source_lang": manifest.get("source_lang"),
                "target_lang": manifest.get("target_lang"),
                "source_chars": len(source_text),
                "hypothesis_chars": len(hypothesis_text),
                "qe_segment_count": parts,
                "qe_segmentation": "short_proportional_text_chunks",
                "qe_max_source_chars": args.max_source_chars,
                "qe_max_hypothesis_chars": args.max_hypothesis_chars,
            }
        )
        for idx, (source_chunk, hypothesis_chunk) in enumerate(zip(source_chunks, hypothesis_chunks)):
            segment_rows.append(
                {
                    "qe_id": f"{key}||seg={idx:03d}",
                    "qe_row_key": key,
                    "run_id": run_id,
                    "model": row_model(row),
                    "chunk_ms": row_chunk_ms(row),
                    "eval_label": row.get("eval_label"),
                    "speed_factor": row.get("speed_factor"),
                    "segment_index": idx,
                    "segment_count": parts,
                    "source": source_chunk,
                    "hypothesis": hypothesis_chunk,
                    "reference": "",
                    "weight_chars": max(1, len(source_chunk) + len(hypothesis_chunk)),
                }
            )
    write_jsonl(args.output_runs, run_rows)
    write_jsonl(args.output_segments, segment_rows)
    print(json.dumps({"runs": len(run_rows), "segments": len(segment_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
