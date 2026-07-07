#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.floras_qe import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate segment-level reference-free QE scores to FLORAS rows.")
    parser.add_argument("--segments-jsonl", required=True)
    parser.add_argument("--runs-jsonl", required=True)
    parser.add_argument("--xcomet-jsonl", default="")
    parser.add_argument("--metricx-jsonl", default="")
    parser.add_argument("--xcomet-model", default="myyycroft/XCOMET-lite")
    parser.add_argument("--metricx-model", default="google/metricx-24-hybrid-large-v2p6-bfloat16")
    parser.add_argument("--output-jsonl", required=True)
    return parser.parse_args()


def weighted_mean(values: list[tuple[float, float]]) -> float | None:
    if not values:
        return None
    weight_sum = sum(weight for _value, weight in values)
    if weight_sum <= 0:
        return statistics.fmean(value for value, _weight in values)
    return sum(value * weight for value, weight in values) / weight_sum


def score_index(path: str, score_key: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    out = {}
    for row in read_jsonl(path):
        if row.get("qe_id") is None:
            continue
        if score_key not in row and "prediction" not in row:
            continue
        out[str(row["qe_id"])] = row
    return out


def main() -> None:
    args = parse_args()
    segments = read_jsonl(args.segments_jsonl)
    runs = {str(row["qe_row_key"]): row for row in read_jsonl(args.runs_jsonl)}
    xcomet = score_index(args.xcomet_jsonl, "xcomet_qe_score")
    metricx = score_index(args.metricx_jsonl, "prediction")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        grouped[str(segment["qe_row_key"])].append(segment)
    out_rows: list[dict[str, Any]] = []
    for key, run in sorted(runs.items()):
        row = dict(run)
        row["qe_reference_free"] = True
        row["qe_inputs"] = "source_transcript+hypothesis_asr"
        row["qe_segmentation"] = run.get("qe_segmentation") or "proportional_text_chunks"
        row["qe_segment_count"] = len(grouped.get(key, []))
        x_values: list[tuple[float, float]] = []
        m_values: list[tuple[float, float]] = []
        x_model = None
        m_model = None
        expected_segments = len(grouped.get(key, []))
        for segment in grouped.get(key, []):
            weight = float(segment.get("weight_chars") or 1.0)
            qe_id = str(segment["qe_id"])
            if qe_id in xcomet:
                x_row = xcomet[qe_id]
                x_values.append((float(x_row["xcomet_qe_score"]), weight))
                x_model = x_row.get("xcomet_qe_model") or x_model
            if qe_id in metricx:
                m_row = metricx[qe_id]
                m_values.append((float(m_row["prediction"]), weight))
                m_model = m_row.get("metricx_qe_model") or m_row.get("model_name_or_path") or m_model
        if args.xcomet_jsonl and len(x_values) != expected_segments:
            raise SystemExit(f"incomplete xCOMET QE for {key}: {len(x_values)}/{expected_segments} segments")
        if args.metricx_jsonl and len(m_values) != expected_segments:
            raise SystemExit(f"incomplete MetricX QE for {key}: {len(m_values)}/{expected_segments} segments")
        x_score = weighted_mean(x_values)
        if x_score is not None:
            row["xcomet_qe_score"] = round(x_score, 6)
            row["xcomet_qe_model"] = x_model or args.xcomet_model
            row["xcomet_qe_segments"] = len(x_values)
        metricx_error = weighted_mean(m_values)
        if metricx_error is not None:
            row["metricx_qe_error"] = round(metricx_error, 6)
            row["metricx_qe_score"] = round(25.0 - metricx_error, 6)
            row["metricx_qe_model"] = m_model or args.metricx_model
            row["metricx_qe_segments"] = len(m_values)
        out_rows.append(row)
    write_jsonl(args.output_jsonl, out_rows)
    print(json.dumps({"rows": len(out_rows), "output": args.output_jsonl}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
