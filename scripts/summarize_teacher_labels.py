#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize teacher label validation JSONL.")
    parser.add_argument("labels", nargs="+", help="Teacher label JSONL files.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return parser.parse_args()


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 4)
    values = sorted(values)
    pos = (len(values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    if low == high:
        return round(values[low], 4)
    frac = pos - low
    return round(values[low] * (1.0 - frac) + values[high] * frac, 4)


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "max": None}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 4),
        "p50": pct(values, 0.5),
        "p90": pct(values, 0.9),
        "max": round(max(values), 4),
    }


def summarize(paths: list[str]) -> dict[str, Any]:
    total = 0
    accepted = 0
    rejected = 0
    reason_counts: Counter[str] = Counter()
    output_units: list[float] = []
    budget_ratios: list[float] = []
    reference_ratios: list[float] = []
    by_file: dict[str, int] = {}

    for path in paths:
        file_count = 0
        for record in read_jsonl(path):
            total += 1
            file_count += 1
            validation = record.get("validation", {})
            if validation.get("accepted"):
                accepted += 1
            else:
                rejected += 1
                for reason in validation.get("reasons", []):
                    reason_counts[str(reason)] += 1
            if validation.get("output_units") is not None:
                output_units.append(float(validation["output_units"]))
            if validation.get("output_over_budget_ratio") is not None:
                budget_ratios.append(float(validation["output_over_budget_ratio"]))
            if validation.get("output_reference_ratio") is not None:
                reference_ratios.append(float(validation["output_reference_ratio"]))
        by_file[path] = file_count

    return {
        "records": total,
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": round(accepted / total, 4) if total else None,
        "rejection_reasons": dict(reason_counts.most_common()),
        "output_units": numeric_summary(output_units),
        "output_over_budget_ratio": numeric_summary(budget_ratios),
        "output_reference_ratio": numeric_summary(reference_ratios),
        "by_file": by_file,
    }


def main() -> None:
    args = parse_args()
    result = summarize(args.labels)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
