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
    parser.add_argument("--output-reference-anchor-segments", default="")
    parser.add_argument(
        "--segmentation-mode",
        choices=("manifest_target_sentences", "short_proportional_text_chunks"),
        default="manifest_target_sentences",
    )
    parser.add_argument("--max-source-chars", type=int, default=220)
    parser.add_argument("--max-hypothesis-chars", type=int, default=160)
    return parser.parse_args()


def clean_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def split_text(text: str, parts: int) -> list[str]:
    text = clean_text(text)
    if parts <= 1 or not text:
        return [text]
    if len(text) < parts:
        chars = list(text)
        return chars + [""] * (parts - len(chars))
    chunks: list[str] = []
    start = 0
    length = len(text)
    min_chunk_chars = max(1, int((length / parts) * 0.35))
    for idx in range(parts - 1):
        remaining_after_cut = parts - idx - 1
        min_cut = start + min_chunk_chars
        max_cut = length - (remaining_after_cut * min_chunk_chars)
        if min_cut > max_cut:
            min_cut = start + 1
            max_cut = length - remaining_after_cut
        target = min(max(round(length * (idx + 1) / parts), min_cut), max_cut)
        window_start = max(min_cut, target - 120)
        window_end = min(max_cut, target + 120)
        cut = target
        best_distance = length
        for match in BOUNDARY_RE.finditer(text, window_start, window_end):
            candidate = match.end()
            if candidate < min_cut or candidate > max_cut:
                continue
            distance = abs(candidate - target)
            if distance < best_distance:
                best_distance = distance
                cut = candidate
        chunks.append(text[start:cut].strip())
        start = cut
    chunks.append(text[start:].strip())
    return chunks


def segment_count(source_text: str, hypothesis_text: str, max_source_chars: int, max_hypothesis_chars: int) -> int:
    source_parts = math.ceil(max(1, len(source_text)) / max_source_chars)
    hypothesis_parts = math.ceil(max(1, len(hypothesis_text)) / max_hypothesis_chars)
    return max(1, source_parts, hypothesis_parts)


def make_segment_rows(
    *,
    key: str,
    run_id: str,
    model: str,
    chunk_ms: int | None,
    eval_label: Any,
    speed_factor: Any,
    source_text: str,
    hypothesis_text: str,
    max_source_chars: int,
    max_hypothesis_chars: int,
) -> list[dict[str, Any]]:
    parts = segment_count(source_text, hypothesis_text, max_source_chars, max_hypothesis_chars)
    source_chunks = split_text(source_text, parts)
    hypothesis_chunks = split_text(hypothesis_text, parts)
    if len(source_chunks) != parts or len(hypothesis_chunks) != parts:
        raise SystemExit(f"bad segmentation for {key}: {len(source_chunks)} vs {len(hypothesis_chunks)}")
    rows: list[dict[str, Any]] = []
    for idx, (source_chunk, hypothesis_chunk) in enumerate(zip(source_chunks, hypothesis_chunks)):
        rows.append(
            {
                "qe_id": f"{key}||seg={idx:03d}",
                "qe_row_key": key,
                "run_id": run_id,
                "model": model,
                "chunk_ms": chunk_ms,
                "eval_label": eval_label,
                "speed_factor": speed_factor,
                "segment_index": idx,
                "segment_count": parts,
                "source": source_chunk,
                "hypothesis": hypothesis_chunk,
                "reference": "",
                "weight_chars": max(1, len(source_chunk) + len(hypothesis_chunk)),
            }
        )
    return rows


def make_segment_rows_from_chunks(
    *,
    key: str,
    run_id: str,
    model: str,
    chunk_ms: int | None,
    eval_label: Any,
    speed_factor: Any,
    source_chunks: list[str],
    hypothesis_chunks: list[str],
) -> list[dict[str, Any]]:
    if len(source_chunks) != len(hypothesis_chunks):
        raise SystemExit(f"bad sentence segmentation for {key}: {len(source_chunks)} vs {len(hypothesis_chunks)}")
    rows: list[dict[str, Any]] = []
    parts = len(source_chunks)
    for idx, (source_chunk, hypothesis_chunk) in enumerate(zip(source_chunks, hypothesis_chunks)):
        rows.append(
            {
                "qe_id": f"{key}||seg={idx:03d}",
                "qe_row_key": key,
                "run_id": run_id,
                "model": model,
                "chunk_ms": chunk_ms,
                "eval_label": eval_label,
                "speed_factor": speed_factor,
                "segment_index": idx,
                "segment_count": parts,
                "source": source_chunk,
                "hypothesis": hypothesis_chunk,
                "reference": "",
                "weight_chars": max(1, len(source_chunk) + len(hypothesis_chunk)),
            }
        )
    return rows


def manifest_sentence_chunks(manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    source_chunks: list[str] = []
    target_chunks: list[str] = []
    for item in manifest.get("target_sentences") or []:
        source = clean_text(item.get("source_sentence"))
        target = clean_text(item.get("target_sentence"))
        if source and target:
            source_chunks.append(source)
            target_chunks.append(target)
    return source_chunks, target_chunks


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
    reference_anchor_rows: list[dict[str, Any]] = []
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
        sentence_source_chunks, sentence_reference_chunks = manifest_sentence_chunks(manifest)
        use_manifest_sentences = args.segmentation_mode == "manifest_target_sentences" and sentence_source_chunks
        if use_manifest_sentences:
            parts = len(sentence_source_chunks)
            qe_segmentation = "manifest_target_sentences"
        else:
            parts = segment_count(source_text, hypothesis_text, args.max_source_chars, args.max_hypothesis_chars)
            qe_segmentation = "short_proportional_text_chunks"
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
                "qe_segmentation": qe_segmentation,
                "qe_max_source_chars": args.max_source_chars,
                "qe_max_hypothesis_chars": args.max_hypothesis_chars,
            }
        )
        if use_manifest_sentences:
            segment_rows.extend(
                make_segment_rows_from_chunks(
                    key=key,
                    run_id=run_id,
                    model=row_model(row),
                    chunk_ms=row_chunk_ms(row),
                    eval_label=row.get("eval_label"),
                    speed_factor=row.get("speed_factor"),
                    source_chunks=sentence_source_chunks,
                    hypothesis_chunks=split_text(hypothesis_text, parts),
                )
            )
        else:
            segment_rows.extend(
                make_segment_rows(
                    key=key,
                    run_id=run_id,
                    model=row_model(row),
                    chunk_ms=row_chunk_ms(row),
                    eval_label=row.get("eval_label"),
                    speed_factor=row.get("speed_factor"),
                    source_text=source_text,
                    hypothesis_text=hypothesis_text,
                    max_source_chars=args.max_source_chars,
                    max_hypothesis_chars=args.max_hypothesis_chars,
                )
            )
        if args.output_reference_anchor_segments and not reference_anchor_rows:
            if use_manifest_sentences and sentence_reference_chunks:
                reference_anchor_rows = make_segment_rows_from_chunks(
                    key="ref_anchor_short",
                    run_id=run_id,
                    model="reference_anchor",
                    chunk_ms=None,
                    eval_label="gpt_reference_anchor",
                    speed_factor=row.get("speed_factor"),
                    source_chunks=sentence_source_chunks,
                    hypothesis_chunks=sentence_reference_chunks,
                )
                continue
            reference_text = clean_text(row.get("reference_text"))
            if not reference_text:
                raise SystemExit(f"missing reference_text for reference anchor in {key}")
            reference_anchor_rows = make_segment_rows(
                key="ref_anchor_short",
                run_id=run_id,
                model="reference_anchor",
                chunk_ms=None,
                eval_label="gpt_reference_anchor",
                speed_factor=row.get("speed_factor"),
                source_text=source_text,
                hypothesis_text=reference_text,
                max_source_chars=args.max_source_chars,
                max_hypothesis_chars=args.max_hypothesis_chars,
            )
    write_jsonl(args.output_runs, run_rows)
    write_jsonl(args.output_segments, segment_rows)
    result = {"runs": len(run_rows), "segments": len(segment_rows)}
    if args.output_reference_anchor_segments:
        write_jsonl(args.output_reference_anchor_segments, reference_anchor_rows)
        result["reference_anchor_segments"] = len(reference_anchor_rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
