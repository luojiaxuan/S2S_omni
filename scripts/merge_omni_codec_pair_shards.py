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

from s2s_omni.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-shard Omni wav/code pair outputs.")
    parser.add_argument("--shard-root", required=True, help="Directory containing shard_*/ outputs.")
    parser.add_argument("--output-dir", required=True, help="Merged split output directory.")
    parser.add_argument("--num-shards", type=int, default=0)
    parser.add_argument("--require-all-shards", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def iter_shard_dirs(root: Path, num_shards: int) -> list[Path]:
    if num_shards > 0:
        return [root / f"shard_{index:04d}" for index in range(num_shards)]
    return sorted(path for path in root.glob("shard_*") if path.is_dir())


def rebase_record_paths(record: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    row = dict(record)
    for key in ["wav_path", "codes_path"]:
        raw = row.get(key)
        if not raw:
            continue
        path = Path(str(raw)).resolve()
        try:
            row[key] = path.relative_to(output_dir.resolve()).as_posix()
        except ValueError:
            row[key] = str(path)
    return row


def main() -> None:
    args = parse_args()
    shard_root = Path(args.shard_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dirs = iter_shard_dirs(shard_root, args.num_shards)
    missing = [path for path in shard_dirs if not (path / "pairs.jsonl").exists()]
    if missing and args.require_all_shards:
        raise SystemExit(
            "missing shard pair files: " + ", ".join(str(path / "pairs.jsonl") for path in missing[:8])
        )

    rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    for shard_dir in shard_dirs:
        pairs_path = shard_dir / "pairs.jsonl"
        rejected_path = shard_dir / "pairs_rejected.jsonl"
        if pairs_path.exists():
            for record in read_jsonl(pairs_path):
                sample_id = str(record.get("id") or "")
                if sample_id in seen_ids:
                    duplicate_ids.append(sample_id)
                    continue
                seen_ids.add(sample_id)
                rows.append(rebase_record_paths(record, output_dir))
        if rejected_path.exists():
            rejected_rows.extend(read_jsonl(rejected_path))

    write_jsonl(output_dir / "pairs.jsonl", rows)
    write_jsonl(output_dir / "pairs_rejected.jsonl", rejected_rows)
    summary = {
        "shard_root": str(shard_root),
        "output_dir": str(output_dir),
        "shards": len(shard_dirs),
        "missing_shards": [str(path) for path in missing],
        "pairs": len(rows),
        "rejected": len(rejected_rows),
        "duplicate_ids": len(duplicate_ids),
        "duplicate_id_examples": duplicate_ids[:20],
    }
    (output_dir / "merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

