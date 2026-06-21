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
from s2s_omni.prompts import SYSTEM_COMPRESSION, build_compression_user_prompt
from s2s_omni.schema import S2SSample
from s2s_omni.style import style_violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge teacher label shards.")
    parser.add_argument("--requests", required=True, help="Original teacher requests JSONL.")
    parser.add_argument("--labels", nargs="+", required=True, help="Teacher label shard JSONL files.")
    parser.add_argument("--output", required=True, help="Merged teacher labels JSONL.")
    parser.add_argument("--sft-output", required=True, help="Merged accepted SFT JSONL.")
    parser.add_argument("--keep-rejected", action="store_true")
    parser.add_argument(
        "--skip-style-recheck",
        action="store_true",
        help="Trust existing validation instead of recomputing spoken-style checks.",
    )
    return parser.parse_args()


def build_sft_record(label_record: dict[str, Any]) -> dict[str, Any]:
    sample = S2SSample.from_dict(label_record["sample"])
    answer = str(label_record["compressed_translation"]).strip()
    sample.compressed_translation = answer
    return {
        "id": sample.id,
        "messages": [
            {"role": "system", "content": SYSTEM_COMPRESSION.strip()},
            {"role": "user", "content": build_compression_user_prompt(sample, include_reference=False)},
            {"role": "assistant", "content": answer},
        ],
    }


def apply_style_recheck(label_record: dict[str, Any]) -> dict[str, Any]:
    sample = S2SSample.from_dict(label_record["sample"])
    answer = str(label_record.get("compressed_translation") or "").strip()
    style_reasons = style_violations(answer, sample.tgt_lang)
    if not style_reasons:
        return label_record

    out = dict(label_record)
    validation = dict(out.get("validation") or {})
    reasons = list(validation.get("reasons") or [])
    for reason in style_reasons:
        if reason not in reasons:
            reasons.append(reason)
    validation["reasons"] = reasons
    validation["accepted"] = False
    out["validation"] = validation
    return out


def main() -> None:
    args = parse_args()
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for path in args.labels:
        for record in read_jsonl(path):
            sample_id = str(record.get("id") or "")
            if not sample_id:
                continue
            if sample_id in by_id:
                duplicate_ids.append(sample_id)
                continue
            by_id[sample_id] = record

    merged: list[dict[str, Any]] = []
    sft_records: list[dict[str, Any]] = []
    missing: list[str] = []
    for request in read_jsonl(args.requests):
        sample_id = str(request.get("id") or "")
        label = by_id.get(sample_id)
        if label is None:
            missing.append(sample_id)
            continue
        if not args.skip_style_recheck:
            label = apply_style_recheck(label)
        merged.append(label)
        accepted = bool(label.get("validation", {}).get("accepted", False))
        if accepted or args.keep_rejected:
            sft_records.append(build_sft_record(label))

    write_jsonl(args.output, merged)
    write_jsonl(args.sft_output, sft_records)
    print(
        json.dumps(
            {
                "label_records": len(merged),
                "sft_records": len(sft_records),
                "missing": len(missing),
                "duplicates_ignored": len(duplicate_ids),
                "output": args.output,
                "sft_output": args.sft_output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if missing[:5]:
        print("missing_preview:", ", ".join(missing[:5]))


if __name__ == "__main__":
    main()
