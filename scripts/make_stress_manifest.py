#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_samples, read_yaml, write_samples
from s2s_omni.streaming import expand_speed_conditions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build speed-stressed S2ST manifest JSONL.")
    parser.add_argument("--input", required=True, help="Input sample JSONL.")
    parser.add_argument("--output", required=True, help="Output manifest JSONL.")
    parser.add_argument(
        "--policy",
        default="configs/compression_policy.yaml",
        help="Compression policy YAML.",
    )
    parser.add_argument(
        "--speed-factors",
        default="1.0,1.35,1.7",
        help="Comma-separated source speed-up factors.",
    )
    parser.add_argument("--max-end-lag-s", type=float, default=None)
    parser.add_argument("--preferred-target-wpm", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = read_yaml(args.policy).get("default", {})
    speed_factors = [float(x.strip()) for x in args.speed_factors.split(",") if x.strip()]
    max_end_lag_s = args.max_end_lag_s
    if max_end_lag_s is None:
        max_end_lag_s = float(policy.get("max_end_lag_s", 1.5))
    preferred_target_wpm = args.preferred_target_wpm
    if preferred_target_wpm is None:
        preferred_target_wpm = float(policy.get("preferred_target_wpm", 165.0))

    samples = read_samples(args.input)
    expanded = expand_speed_conditions(
        samples,
        speed_factors=speed_factors,
        max_end_lag_s=max_end_lag_s,
        preferred_target_wpm=preferred_target_wpm,
    )
    write_samples(args.output, expanded)
    print(f"wrote {len(expanded)} stress records to {Path(args.output)}")


if __name__ == "__main__":
    main()
