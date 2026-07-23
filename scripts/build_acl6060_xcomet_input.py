#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ACL6060 segment JSONL for XCOMET-XL.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--source-text-file", type=Path, default=None)
    parser.add_argument("--resegmented-jsonl", type=Path, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def read_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def resolve_source_text(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.source_text_file is not None:
        return args.source_text_file
    path = Path(str(config.get("source_text_file") or ""))
    if path.exists():
        return path
    dataset_root = Path(str(config.get("dataset_root") or ""))
    if dataset_root and "main_result/" in path.as_posix():
        rel = path.as_posix().split("main_result/", 1)[1]
        candidate = dataset_root / "main_result" / rel
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"source_text_file not found for {args.run_dir}")


def run_key(run_dir: Path, config: dict[str, Any]) -> str:
    provider = str(config.get("provider") or "")
    target_lang = str(config.get("target_lang") or "")
    chunk_ms = str(config.get("chunk_ms") or "")
    speed = str(config.get("speed_factor") or "")
    return "||".join([run_dir.name, provider, target_lang, chunk_ms, speed])


def main() -> None:
    args = parse_args()
    config = read_config(args.run_dir)
    resegmented = args.resegmented_jsonl or (
        args.run_dir / "omnisteval_longform" / "instances.resegmented.jsonl"
    )
    source_text_file = resolve_source_text(args, config)
    source_lines = source_text_file.read_text(encoding="utf-8").splitlines()
    segments = read_jsonl(resegmented)
    if len(source_lines) != len(segments):
        raise ValueError(
            f"source/resegmented row mismatch: {len(source_lines)} vs {len(segments)}"
        )
    key = run_key(args.run_dir, config)
    rows = []
    for segment, source in zip(segments, source_lines):
        index = int(segment.get("index") or 0)
        hypothesis = str(segment.get("prediction") or "")
        reference = str(segment.get("reference") or "")
        rows.append(
            {
                "xcomet_id": f"{key}||{index:04d}",
                "run_key": key,
                "run_dir": str(args.run_dir),
                "segment_index": index,
                "source": source,
                "hypothesis": hypothesis,
                "reference": reference,
                "weight_chars": max(1, len(reference)),
                "target_lang": config.get("target_lang"),
                "provider": config.get("provider"),
                "chunk_ms": config.get("chunk_ms"),
                "speed_factor": config.get("speed_factor"),
            }
        )
    write_jsonl(args.output_jsonl, rows)
    print(json.dumps({"rows": len(rows), "output": str(args.output_jsonl)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
