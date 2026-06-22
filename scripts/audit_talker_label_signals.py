#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl, write_jsonl
from s2s_omni.schema import S2SSample
from s2s_omni.speech_signals import coverage_signals, duration_signals, gate_reasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit talker code labels with duration, ASR coverage, and style signals."
    )
    parser.add_argument("--input", required=True, help="talker_code_labels.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument(
        "--asr-jsonl",
        help="Optional JSONL keyed by id with asr_text/transcript/text for generated speech.",
    )
    parser.add_argument("--max-duration-budget-ratio", type=float, default=1.05)
    parser.add_argument("--min-coverage-recall", type=float, default=0.9)
    parser.add_argument("--min-last-matched-ratio", type=float, default=0.9)
    parser.add_argument("--max-rate-pressure-ratio", type=float, default=1.35)
    return parser.parse_args()


def load_asr(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    out: dict[str, str] = {}
    for record in read_jsonl(path):
        sample_id = str(record.get("id") or "")
        text = record.get("asr_text") or record.get("transcript") or record.get("text")
        if sample_id and text:
            out[sample_id] = str(text)
    return out


def audit_record(record: dict[str, Any], asr_by_id: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    sample = S2SSample.from_dict(record["sample"])
    target_text = str(record.get("target_text") or sample.preferred_target_text())
    speech_duration_s = record.get("codec_duration_s") or record.get("wav_duration_s")
    duration = duration_signals(sample, target_text, speech_duration_s)
    asr_text = asr_by_id.get(sample.id)
    coverage = coverage_signals(target_text, asr_text, sample.tgt_lang) if asr_text is not None else None
    reasons = gate_reasons(
        duration,
        coverage,
        max_duration_budget_ratio=args.max_duration_budget_ratio,
        min_coverage_recall=args.min_coverage_recall,
        min_last_matched_ratio=args.min_last_matched_ratio,
        max_rate_pressure_ratio=args.max_rate_pressure_ratio,
    )
    return {
        **record,
        "speech_duration_signals": duration,
        "speech_coverage_signals": coverage,
        "speech_label_gate_pass": not reasons,
        "speech_label_gate_reasons": reasons,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    passed = 0
    duration_ratios = []
    coverage_recalls = []
    for row in rows:
        if row.get("speech_label_gate_pass"):
            passed += 1
        for reason in row.get("speech_label_gate_reasons") or []:
            reason_counts[reason.split(":", 1)[0]] += 1
        duration = row.get("speech_duration_signals") or {}
        if duration.get("duration_budget_ratio") is not None:
            duration_ratios.append(float(duration["duration_budget_ratio"]))
        coverage = row.get("speech_coverage_signals") or {}
        if coverage.get("target_coverage_recall") is not None:
            coverage_recalls.append(float(coverage["target_coverage_recall"]))
    return {
        "records": len(rows),
        "gate_passed": passed,
        "gate_failed": len(rows) - passed,
        "reason_counts": dict(reason_counts),
        "duration_budget_ratio": _summary_stats(duration_ratios),
        "target_coverage_recall": _summary_stats(coverage_recalls),
    }


def _summary_stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    values = sorted(values)
    return {
        "min": round(values[0], 6),
        "p50": round(values[len(values) // 2], 6),
        "p90": round(values[min(len(values) - 1, int(len(values) * 0.9))], 6),
        "max": round(values[-1], 6),
    }


def main() -> None:
    args = parse_args()
    asr_by_id = load_asr(args.asr_jsonl)
    rows = [audit_record(record, asr_by_id, args) for record in read_jsonl(args.input)]
    write_jsonl(args.output, rows)
    summary = summarize(rows)
    summary_path = args.summary_output or str(Path(args.output).with_suffix(".summary.json"))
    Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
