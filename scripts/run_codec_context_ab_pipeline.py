#!/usr/bin/env python3
"""Run frozen-code generation and codec-context A/B from one JSON config."""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def run_script(path: Path, arguments: list[str]) -> None:
    previous_argv = sys.argv
    try:
        sys.argv = [str(path), *arguments]
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = previous_argv


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text())
    run_script(Path(config["generate_script"]), config["generate_arguments"])
    run_script(Path(config["evaluate_script"]), config["evaluate_arguments"])


if __name__ == "__main__":
    main()
