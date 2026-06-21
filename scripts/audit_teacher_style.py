#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl
from s2s_omni.style import style_violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SFT/teacher labels for spoken-style violations.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def candidate_text(record: dict) -> tuple[str, str, str]:
    lang = "zh"
    sample_id = str(record.get("id") or record.get("sample", {}).get("id") or "")
    if record.get("compressed_translation"):
        sample = record.get("sample") or {}
        return sample_id, str(sample.get("tgt_lang") or lang), str(record["compressed_translation"])
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        return sample_id, lang, str(messages[-1].get("content") or "")
    sample = record.get("sample") or {}
    return sample_id, str(sample.get("tgt_lang") or lang), str(sample.get("compressed_translation") or "")


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    bad = []
    reason_counts: dict[str, int] = {}
    for record in rows:
        sample_id, lang, text = candidate_text(record)
        reasons = style_violations(text, lang)
        if not reasons:
            continue
        bad.append((sample_id, reasons, text))
        for reason in reasons:
            key = reason.split(":", 1)[0]
            reason_counts[key] = reason_counts.get(key, 0) + 1
    print(
        json.dumps(
            {
                "input": args.input,
                "total": len(rows),
                "violations": len(bad),
                "violation_rate": round(len(bad) / len(rows), 4) if rows else 0,
                "reason_counts": reason_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for sample_id, reasons, text in bad[: args.limit]:
        print(json.dumps({"id": sample_id, "reasons": reasons, "text": text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
