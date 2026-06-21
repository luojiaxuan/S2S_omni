#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_samples, write_jsonl
from s2s_omni.sft import iter_sft_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export compression-aware SFT JSONL.")
    parser.add_argument("--input", required=True, help="Input manifest JSONL.")
    parser.add_argument("--output", required=True, help="Output SFT JSONL.")
    parser.add_argument(
        "--format",
        default="messages",
        choices=["messages", "prompt_completion"],
        help="Training record format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_samples(args.input)
    records = list(iter_sft_records(samples, format_name=args.format))
    write_jsonl(args.output, records)
    print(f"wrote {len(records)} SFT records to {Path(args.output)}")


if __name__ == "__main__":
    main()
