#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that JSONL splits do not share base_id values."
    )
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        help="Named split in the form name=path. Repeat for train/dev/test.",
    )
    parser.add_argument("--output", help="Optional JSON summary path.")
    return parser.parse_args()


def parse_named_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"--split must be name=path, got: {value}")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"--split must be name=path, got: {value}")
    return name, path


def record_payload(record: dict[str, Any]) -> dict[str, Any]:
    sample = record.get("sample")
    if isinstance(sample, dict):
        return sample
    return record


def base_id_from_record(record: dict[str, Any]) -> str:
    payload = record_payload(record)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    base_id = metadata.get("base_id")
    if base_id:
        return str(base_id)
    sample_id = str(payload.get("id") or record.get("id") or record.get("sample_id") or "")
    return sample_id.split("__speed_", 1)[0]


def load_base_ids(path: str) -> set[str]:
    base_ids: set[str] = set()
    for record in read_jsonl(path):
        base_id = base_id_from_record(record)
        if base_id:
            base_ids.add(base_id)
    return base_ids


def main() -> None:
    args = parse_args()
    splits = [parse_named_path(value) for value in args.split]
    base_ids_by_split = {name: load_base_ids(path) for name, path in splits}

    overlaps: dict[str, dict[str, Any]] = {}
    names = list(base_ids_by_split)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sorted(base_ids_by_split[left] & base_ids_by_split[right])
            overlaps[f"{left}__{right}"] = {
                "count": len(overlap),
                "examples": overlap[:10],
            }

    summary = {
        "splits": {
            name: {"path": path, "base_ids": len(base_ids_by_split[name])}
            for name, path in splits
        },
        "overlaps": overlaps,
        "passed": all(item["count"] == 0 for item in overlaps.values()),
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
