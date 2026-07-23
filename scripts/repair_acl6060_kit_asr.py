#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KIT_EVAL_PATH = ROOT / "scripts" / "run_acl6060_kit_live_eval.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace truncated full-wav KIT ASR with grouped-window ASR."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--api-key-file", required=True, type=Path)
    parser.add_argument("--asr-window-s", type=float, default=120.0)
    parser.add_argument("--index", type=int, action="append", default=[])
    return parser.parse_args()


def load_kit_eval() -> Any:
    spec = importlib.util.spec_from_file_location("acl6060_kit_eval", KIT_EVAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {KIT_EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    fd, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def sample_dir(run_dir: Path, row: dict[str, Any]) -> Path:
    source = str((row.get("source") or [""])[0])
    candidate = run_dir / f"{int(row['index']):03d}_{Path(source).stem}"
    if not candidate.is_dir():
        raise FileNotFoundError(candidate)
    return candidate


def main() -> None:
    args = parse_args()
    module = load_kit_eval()
    run_config_path = args.run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    if run_config.get("provider") != "kit":
        raise ValueError(f"not a KIT run: {args.run_dir}")
    api_key = module.read_secret(args.api_key_file)
    asr_args = SimpleNamespace(
        resume=True,
        asr_base_url=str(run_config.get("asr_base_url") or "https://api.openai.com/v1"),
        asr_model=str(run_config.get("asr_model") or "gpt-4o-mini-transcribe"),
        asr_window_s=args.asr_window_s,
    )

    instances_path = args.run_dir / "instances.log"
    responses_path = args.run_dir / "responses.jsonl"
    instances = read_jsonl(instances_path)
    responses = read_jsonl(responses_path)
    responses_by_index = {int(row["index"]): row for row in responses}
    selected = set(args.index)
    repaired = 0
    for row in instances:
        index = int(row["index"])
        if selected and index not in selected:
            continue
        paths = module.sample_result_paths(sample_dir(args.run_dir, row))
        asr = module.transcribe_target(
            asr_args,
            api_key,
            paths["target_wav"],
            paths["chunks_jsonl"],
            paths["asr_json"],
            paths["asr_windows_jsonl"],
        )
        prediction = str(asr.get("asr_text") or "").strip()
        delays, elapsed, timing = module.target_unit_times_ms(
            text=prediction,
            target_lang=str(run_config["target_lang"]),
            chunks_jsonl=paths["chunks_jsonl"],
            source_length_ms=float(row["source_length"]),
        )
        if not delays:
            raise RuntimeError(f"empty grouped-window ASR: {paths['target_wav']}")
        row["prediction"] = prediction
        row["delays"] = delays
        row["elapsed"] = elapsed
        row["prediction_length"] = len(delays)
        response = responses_by_index.get(index)
        if response is not None:
            response["prediction_units"] = len(delays)
            response["asr_strategy"] = asr["asr_strategy"]
            response["asr_window_count"] = asr["asr_window_count"]
            response.update(timing)
        repaired += 1

    for path in [instances_path, responses_path]:
        backup = path.with_name(f"{path.name}.pre_windowed_asr")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
    atomic_write_jsonl(instances_path, instances)
    atomic_write_jsonl(responses_path, responses)
    run_config["asr_strategy"] = "tts_chunk_grouped_windows_v1"
    run_config["asr_window_s"] = args.asr_window_s
    atomic_write_json(run_config_path, run_config)
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "repaired_rows": repaired,
                "asr_window_s": args.asr_window_s,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
