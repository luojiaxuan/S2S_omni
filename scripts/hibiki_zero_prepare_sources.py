#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.hibiki_zero import (
    HibikiSample,
    duration_budgets_from_source,
    normalize_src_lang,
)
from s2s_omni.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize X->English source rows into the Hibiki-Zero data schema."
    )
    parser.add_argument("--input", action="append", required=True, help="Input JSONL. Can repeat.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", default="source_manifest.jsonl")
    parser.add_argument("--languages", default="fr,es,pt,de")
    parser.add_argument("--max-per-lang", type=int, default=0)
    parser.add_argument("--duration-rtf-threshold", type=float, default=1.0)
    parser.add_argument("--duration-slack-s", type=float, default=0.0)
    parser.add_argument("--split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--dev-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def iter_samples(paths: Iterable[str]) -> Iterable[tuple[str, int, dict[str, Any]]]:
    for path in paths:
        for idx, record in enumerate(read_jsonl(path)):
            yield path, idx, record


def normalize_record(record: dict[str, Any], path: str, index: int, args: argparse.Namespace) -> dict[str, Any]:
    data = dict(record)
    if "sample_id" not in data and "id" not in data and "utterance_id" not in data:
        data["sample_id"] = f"{Path(path).stem}_{index:08d}"
    sample = HibikiSample.from_dict(data)
    source_durations = sample.source_durations_s()
    budgets = duration_budgets_from_source(
        source_durations,
        rtf_threshold=args.duration_rtf_threshold,
        slack_s=args.duration_slack_s,
    )
    metadata = dict(sample.metadata)
    metadata.update(
        {
            "source_manifest_path": path,
            "source_manifest_index": index,
            "duration_rtf_threshold": args.duration_rtf_threshold,
            "duration_slack_s": args.duration_slack_s,
        }
    )
    normalized = sample.to_dict()
    normalized["duration_budget_s"] = budgets
    normalized["metadata"] = metadata
    normalized["source_audio_chunks"] = [
        {
            **chunk.to_dict(),
            "source_duration_s": source_durations[idx],
            "duration_budget_s": budgets[idx],
        }
        for idx, chunk in enumerate(sample.source_audio_chunks)
    ]
    return normalized


def split_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(args.seed)
    by_id = {str(row["sample_id"]): row for row in rows}
    ids = sorted(by_id)
    rng.shuffle(ids)
    n_train = int(round(len(ids) * args.train_ratio))
    n_dev = int(round(len(ids) * args.dev_ratio))
    train_ids = set(ids[:n_train])
    dev_ids = set(ids[n_train : n_train + n_dev])
    test_ids = set(ids[n_train + n_dev :])
    return {
        "train": [by_id[key] for key in ids if key in train_ids],
        "dev": [by_id[key] for key in ids if key in dev_ids],
        "test": [by_id[key] for key in ids if key in test_ids],
    }


def main() -> None:
    args = parse_args()
    languages = {normalize_src_lang(item) for item in args.languages.split(",") if item.strip()}
    counts: dict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for path, index, record in iter_samples(args.input):
        try:
            src_lang = normalize_src_lang(str(record.get("src_lang") or record.get("source_language") or ""))
            if src_lang not in languages:
                continue
            if args.max_per_lang > 0 and counts[src_lang] >= args.max_per_lang:
                continue
            row = normalize_record(record, path, index, args)
            rows.append(row)
            counts[src_lang] += 1
        except Exception as exc:
            rejected.append(
                {
                    "source_manifest_path": path,
                    "source_manifest_index": index,
                    "reject_reasons": [f"exception:{type(exc).__name__}"],
                    "error": str(exc),
                }
            )

    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output
    write_jsonl(output_path, rows)
    if rejected:
        write_jsonl(output_dir / "source_manifest_rejected.jsonl", rejected)
    split_summary: dict[str, int] = {}
    if args.split:
        splits = split_rows(rows, args)
        for name, split_rows_value in splits.items():
            write_jsonl(output_dir / f"{name}.jsonl", split_rows_value)
            split_summary[name] = len(split_rows_value)
    summary = {
        "inputs": args.input,
        "output": str(output_path),
        "rows": len(rows),
        "rejected": len(rejected),
        "languages": sorted(languages),
        "counts_by_lang": dict(sorted(counts.items())),
        "split": split_summary,
    }
    (output_dir / "prepare_sources_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
