#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl
from s2s_omni.metrics import optional_sacrebleu
from s2s_omni.rasst import audio_duration_s
from s2s_omni.speech_signals import coverage_signals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Hibiki-Zero generated speech predictions.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def key(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("id"))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"count": len(rows)}
    for field in [
        "generated_duration_s",
        "target_duration_s",
        "duration_ratio",
        "speech_s2s_rtf_full",
        "target_coverage_recall",
    ]:
        vals = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
        if vals:
            summary[field] = {
                "mean": round(statistics.fmean(vals), 6),
                "p50": percentile(vals, 0.5),
                "p90": percentile(vals, 0.9),
            }
    for field in ["missing_prediction", "empty_output", "speech_s2s_rtf_violation"]:
        vals = [bool(row.get(field)) for row in rows if field in row]
        if vals:
            summary[f"{field}_rate"] = round(sum(vals) / len(vals), 6)
    refs = [str(row.get("compressed_en_text") or "") for row in rows if row.get("asr_text")]
    hyps = [str(row.get("asr_text") or "") for row in rows if row.get("asr_text")]
    if hyps and refs:
        summary["asr_text_metrics"] = optional_sacrebleu(hyps, refs)
    return summary


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return round(values[0], 6)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return round(values[lo] * (1 - frac) + values[hi] * frac, 6)


def main() -> None:
    args = parse_args()
    manifest = {key(row): row for row in read_jsonl(args.manifest)}
    predictions = {key(row): row for row in read_jsonl(args.predictions)}
    rows = []
    for sample_id, gold in manifest.items():
        row = dict(gold)
        pred = predictions.get(sample_id)
        if not pred:
            row["missing_prediction"] = True
            rows.append(row)
            continue
        row.update(
            {
                "missing_prediction": False,
                "generated_wav_path": pred.get("generated_wav_path") or pred.get("wav_path"),
                "asr_text": pred.get("asr_text") or pred.get("prediction_text") or "",
            }
        )
        if row.get("generated_wav_path"):
            row["generated_duration_s"] = round(audio_duration_s(row["generated_wav_path"]), 6)
            row["empty_output"] = row["generated_duration_s"] < 0.08
        if row.get("target_en_wav"):
            row["target_duration_s"] = round(audio_duration_s(row["target_en_wav"]), 6)
        if row.get("generated_duration_s") and row.get("target_duration_s"):
            row["duration_ratio"] = round(row["generated_duration_s"] / row["target_duration_s"], 6)
        source_durations = [
            chunk.get("source_duration_s") for chunk in row.get("source_audio_chunks") or []
        ]
        source_total = sum(float(value or 0.0) for value in source_durations)
        if row.get("generated_duration_s") and source_total > 0:
            row["speech_s2s_rtf_full"] = round(row["generated_duration_s"] / source_total, 6)
            row["speech_s2s_rtf_violation"] = row["speech_s2s_rtf_full"] > 1.0
        if row.get("asr_text"):
            row.update(coverage_signals(row.get("compressed_en_text") or "", row["asr_text"], "en"))
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
    summary = summarize(rows)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
