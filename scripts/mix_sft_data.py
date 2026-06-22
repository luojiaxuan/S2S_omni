#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix primary and auxiliary SFT JSONL files.")
    parser.add_argument("--primary", required=True, help="Primary SFT JSONL, e.g. compression labels.")
    parser.add_argument(
        "--aux",
        nargs="+",
        required=True,
        help="One or more auxiliary SFT JSONL files, e.g. pass-through and faithful warm-up.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--aux-limit", type=int, default=0, help="0 means use all auxiliary records.")
    parser.add_argument("--primary-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def tag(record: dict[str, Any], source: str) -> dict[str, Any]:
    out = dict(record)
    metadata = dict(out.get("metadata") or {})
    metadata["mix_source"] = source
    out["metadata"] = metadata
    return out


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    primary = [tag(record, "primary") for record in read_jsonl(args.primary)]
    aux = []
    aux_counts: dict[str, int] = {}
    for aux_path in args.aux:
        source = f"aux:{Path(aux_path).stem}"
        records = [tag(record, source) for record in read_jsonl(aux_path)]
        aux_counts[aux_path] = len(records)
        aux.extend(records)
    if args.aux_limit > 0 and len(aux) > args.aux_limit:
        aux = rng.sample(aux, args.aux_limit)
    mixed = primary * max(1, args.primary_repeat) + aux
    rng.shuffle(mixed)
    write_jsonl(args.output, mixed)
    print(
        json.dumps(
            {
                "primary": len(primary),
                "primary_repeat": max(1, args.primary_repeat),
                "aux": len(aux),
                "aux_inputs": aux_counts,
                "mixed": len(mixed),
                "output": args.output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
