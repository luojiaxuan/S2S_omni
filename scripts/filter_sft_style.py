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
from s2s_omni.style import style_violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter SFT JSONL by spoken-style checks.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reject-output")
    return parser.parse_args()


def candidate_text(record: dict[str, Any]) -> tuple[str, str, str]:
    sample_id = str(record.get("id") or record.get("sample", {}).get("id") or "")
    sample = record.get("sample") if isinstance(record.get("sample"), dict) else {}
    lang = str(sample.get("tgt_lang") or record.get("tgt_lang") or "zh")

    if record.get("compressed_translation"):
        return sample_id, lang, str(record["compressed_translation"])

    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        return sample_id, lang, str(messages[-1].get("content") or "")

    return sample_id, lang, str(sample.get("compressed_translation") or "")


def main() -> None:
    args = parse_args()
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}

    for record in read_jsonl(args.input):
        sample_id, lang, text = candidate_text(record)
        reasons = style_violations(text, lang)
        if reasons:
            marked = dict(record)
            metadata = dict(marked.get("metadata") or {})
            metadata["style_reject_reasons"] = reasons
            marked["metadata"] = metadata
            rejected.append(marked)
            for reason in reasons:
                key = reason.split(":", 1)[0]
                reason_counts[key] = reason_counts.get(key, 0) + 1
            continue
        kept.append(record)

    write_jsonl(args.output, kept)
    if args.reject_output:
        write_jsonl(args.reject_output, rejected)

    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "reject_output": args.reject_output,
                "kept": len(kept),
                "rejected": len(rejected),
                "total": len(kept) + len(rejected),
                "reason_counts": reason_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
