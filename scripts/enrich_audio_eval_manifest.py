#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge talker-label audit signals into an audio eval manifest."
    )
    parser.add_argument("--manifest", required=True, help="Audio eval manifest JSONL keyed by id.")
    parser.add_argument("--audit-jsonl", required=True, help="Output from audit_talker_label_signals.py.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix", default="evo", help="Manifest prefix to enrich, usually evo or base.")
    return parser.parse_args()


def flatten_audit(record: dict[str, Any], prefix: str) -> dict[str, Any]:
    duration = record.get("speech_duration_signals") or {}
    coverage = record.get("speech_coverage_signals") or {}
    flat = {
        f"{prefix}_speech_label_gate_pass": record.get("speech_label_gate_pass"),
        f"{prefix}_speech_label_gate_reasons": record.get("speech_label_gate_reasons") or [],
    }
    for key in [
        "target_units",
        "speech_duration_s",
        "budget_duration_s",
        "reference_tts_duration_s",
        "source_wall_duration_s",
        "style_violations",
        "target_units_per_s",
        "target_units_per_min",
        "duration_budget_ratio",
        "duration_reference_tts_ratio",
        "speech_s2s_rtf",
        "estimated_default_text_duration_s",
        "rate_pressure_ratio",
    ]:
        if key in duration:
            flat[f"{prefix}_{key}"] = duration[key]
    for key in [
        "asr_text",
        "target_coverage_units",
        "target_coverage_recall",
        "asr_precision_vs_target",
        "ordered_target_recall",
        "last_matched_target_unit_ratio",
        "unmatched_target_suffix_units",
        "bag_target_recall",
        "bag_asr_precision",
    ]:
        if key in coverage:
            flat[f"{prefix}_{key}"] = coverage[key]
    return flat


def main() -> None:
    args = parse_args()
    audits = {
        str(record.get("id") or ""): flatten_audit(record, args.prefix)
        for record in read_jsonl(args.audit_jsonl)
        if record.get("id")
    }
    rows = []
    for row in read_jsonl(args.manifest):
        sample_id = str(row.get("id") or "")
        rows.append({**row, **audits.get(sample_id, {})})
    write_jsonl(args.output, rows)
    enriched = sum(1 for row in rows if str(row.get("id") or "") in audits)
    print(f"wrote {args.output} with {enriched} enriched rows")


if __name__ == "__main__":
    main()
