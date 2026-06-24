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
from s2s_omni.rasst import audio_duration_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Hibiki-Zero sample or turn manifests.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dev-manifest", default="")
    parser.add_argument("--test-manifest", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-duration-sum-delta", type=float, default=0.35)
    return parser.parse_args()


def base_id(row: dict[str, Any]) -> str:
    return str(row.get("base_sample_id") or row.get("sample_id") or row.get("id") or "")


def validate_rows(rows: list[dict[str, Any]], max_delta: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    issues = []
    rtf_values = []
    corrupt_wavs = 0
    missing_wavs = 0
    for row in rows:
        reasons = []
        target_wavs = row.get("target_chunk_wavs") or []
        boundaries = row.get("mfa_boundaries") or []
        chunk_durations = row.get("target_chunk_duration_s") or []
        if target_wavs and len(target_wavs) != len(row.get("source_audio_chunks") or []):
            reasons.append("target_chunk_count_mismatch")
        if boundaries and len(boundaries) != len(row.get("source_audio_chunks") or []):
            reasons.append("boundary_count_mismatch")
        for path in target_wavs:
            if not path:
                continue
            wav_path = Path(path)
            if not wav_path.exists():
                missing_wavs += 1
                reasons.append("missing_target_wav")
                continue
            try:
                audio_duration_s(wav_path)
            except Exception:
                corrupt_wavs += 1
                reasons.append("corrupt_target_wav")
        target_full = row.get("target_en_wav")
        if target_full and Path(str(target_full)).exists() and chunk_durations:
            full_s = audio_duration_s(target_full)
            chunk_sum = sum(float(x or 0) for x in chunk_durations)
            if abs(full_s - chunk_sum) > max_delta:
                reasons.append("target_chunk_duration_sum_delta")
        for value in row.get("speech_s2s_rtf") or []:
            if isinstance(value, (int, float)):
                rtf_values.append(float(value))
        if reasons:
            issues.append({"sample_id": row.get("sample_id"), "issues": sorted(set(reasons))})
    summary: dict[str, Any] = {
        "rows": len(rows),
        "issues": len(issues),
        "missing_wavs": missing_wavs,
        "corrupt_wavs": corrupt_wavs,
    }
    if rtf_values:
        summary["speech_s2s_rtf"] = {
            "mean": round(statistics.fmean(rtf_values), 6),
            "p50": percentile(rtf_values, 0.5),
            "p90": percentile(rtf_values, 0.9),
            "violation_rate": round(sum(v > 1.0 for v in rtf_values) / len(rtf_values), 6),
        }
    return issues, summary


def split_overlap(paths: list[str]) -> dict[str, Any]:
    sets = []
    for path in paths:
        if not path:
            sets.append(set())
            continue
        sets.append({base_id(row) for row in read_jsonl(path) if base_id(row)})
    names = ["train", "dev", "test"]
    overlaps = {}
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            overlaps[f"{names[i]}_{names[j]}"] = len(overlap)
    return {"sizes": dict(zip(names, [len(x) for x in sets], strict=False)), "overlaps": overlaps}


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
    rows = read_jsonl(args.manifest)
    issues, summary = validate_rows(rows, args.max_duration_sum_delta)
    if args.dev_manifest or args.test_manifest:
        summary["split_integrity"] = split_overlap(
            [args.manifest, args.dev_manifest, args.test_manifest]
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summary, "issues": issues[:1000]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["missing_wavs"] or summary["corrupt_wavs"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
