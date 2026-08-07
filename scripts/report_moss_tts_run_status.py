#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report MOSS-TTS InfiniSST run status.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    return parser.parse_args()


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def has_spoken_content(text: str) -> bool:
    return any(unicodedata.category(ch)[0] in {"L", "N"} for ch in text)


def expected_counts(dataset_root: Path, split: str) -> dict[str, Any]:
    path = dataset_root / "manifest" / f"{split}_moss_requests.jsonl"
    total = spoken = punct_only = empty = 0
    for row in read_jsonl(path):
        total += 1
        text = str(row.get("target_text") or "").strip()
        if not text:
            empty += 1
        elif has_spoken_content(text):
            spoken += 1
        else:
            punct_only += 1
    return {
        "manifest": str(path),
        "total": total,
        "spoken": spoken,
        "punct_only": punct_only,
        "empty": empty,
    }


def split_counts(run_root: Path, split: str) -> dict[str, Any]:
    accepted_ids: list[str] = []
    rejected_ids: list[str] = []
    bad_rejects: list[dict[str, Any]] = []
    shard_counts = []
    for shard in range(4):
        raw_path = run_root / "raw" / f"{split}_moss_raw_shard{shard}.jsonl"
        rejected_path = run_root / "raw" / f"{split}_moss_rejected_shard{shard}.jsonl"
        raw_rows = list(read_jsonl(raw_path) or [])
        rejected_rows = list(read_jsonl(rejected_path) or [])
        accepted_ids.extend(str(row.get("id")) for row in raw_rows)
        for row in rejected_rows:
            sample_id = str(row.get("id"))
            rejected_ids.append(sample_id)
            text = str((row.get("row") or {}).get("target_text") or "").strip()
            if has_spoken_content(text):
                bad_rejects.append(
                    {
                        "id": sample_id,
                        "target_text": text,
                        "error": row.get("error"),
                    }
                )
        shard_counts.append(
            {
                "shard": shard,
                "accepted": len(raw_rows),
                "rejected": len(rejected_rows),
                "raw_path": str(raw_path),
                "rejected_path": str(rejected_path),
            }
        )

    accepted_set = set(accepted_ids)
    rejected_set = set(rejected_ids)
    id_counts = Counter(accepted_ids + rejected_ids)
    duplicates = [
        sample_id
        for sample_id, count in sorted(id_counts.items())
        if count > 1
    ]
    return {
        "accepted": len(accepted_ids),
        "rejected": len(rejected_ids),
        "covered": len(accepted_set | rejected_set),
        "overlap": sorted(accepted_set & rejected_set)[:20],
        "duplicates": duplicates[:20],
        "bad_rejects": bad_rejects[:20],
        "bad_reject_count": len(bad_rejects),
        "shards": shard_counts,
    }


def pid_status(run_root: Path) -> list[dict[str, Any]]:
    statuses = []
    for pidfile in sorted((run_root / "pids").glob("*.pid")):
        try:
            pid = int(pidfile.read_text().strip())
        except Exception as exc:
            statuses.append({"pidfile": str(pidfile), "error": str(exc), "alive": False})
            continue
        alive = True
        try:
            os.kill(pid, 0)
        except OSError:
            alive = False
        statuses.append({"pidfile": str(pidfile), "pid": pid, "alive": alive})
    return statuses


def artifact_status(run_root: Path) -> dict[str, Any]:
    prepared = sorted(glob.glob(str(run_root / "prepared" / "*.jsonl")))
    checkpoints = sorted(glob.glob(str(run_root / "checkpoints" / "**" / "model.safetensors"), recursive=True))
    logs = {
        "prepare_train_supervisor": str(run_root / "logs" / "99_prepare_train_supervisor.log"),
        "train": str(run_root / "logs" / "03_train.log"),
    }
    return {
        "prepared_jsonl": prepared,
        "checkpoint_models": checkpoints,
        "logs": logs,
    }


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    dataset_root = Path(args.dataset_root)
    report = {
        "run_root": str(run_root),
        "dataset_root": str(dataset_root),
        "splits": {},
        "pids": pid_status(run_root),
        "artifacts": artifact_status(run_root),
    }
    for split in ("train", "dev"):
        expected = expected_counts(dataset_root, split)
        actual = split_counts(run_root, split)
        missing_or_pending = max(0, expected["total"] - actual["covered"])
        report["splits"][split] = {
            "expected": expected,
            "actual": actual,
            "missing_or_pending": missing_or_pending,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
