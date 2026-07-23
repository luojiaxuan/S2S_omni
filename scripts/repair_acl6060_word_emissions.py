#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LIVE_EVAL_PATH = (
    ROOT
    / "projects"
    / "acl6060_s2s_metrics_seed"
    / "run_acl6060_live_stream_eval.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair word-level ACL6060 latency emissions from raw API events."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--provider", choices=["openai", "gemini"], default="")
    return parser.parse_args()


def load_live_eval() -> Any:
    spec = importlib.util.spec_from_file_location("acl6060_live_eval", LIVE_EVAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {LIVE_EVAL_PATH}")
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


def find_sample_dir(run_dir: Path, index: int, source: str) -> Path:
    stem = Path(source).stem
    matches = sorted(run_dir.glob(f"{index:03d}_{stem}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one sample directory for index={index}, source={stem}: {matches}"
        )
    return matches[0]


def event_fragment(module: Any, event: dict[str, Any], provider: str) -> str:
    if provider == "openai":
        return module.openai_output_delta(event)
    return module.gemini_transcription(event, "outputTranscription")


def repaired_emissions(
    module: Any,
    raw_events_path: Path,
    provider: str,
    expected_prediction: str,
) -> tuple[list[float], list[float]]:
    streamed = module.StreamedText([], [], [], [], [])
    for event in read_jsonl(raw_events_path):
        fragment = event_fragment(module, event, provider)
        if not fragment:
            continue
        module.add_text_delta(
            streamed,
            fragment,
            "de",
            float(event.get("sent_source_ms") or 0.0),
            float(event.get("received_at_s") or 0.0) * 1000.0,
        )
    actual_prediction = "".join(streamed.text_parts).strip()
    if actual_prediction != expected_prediction.strip():
        raise RuntimeError(
            f"raw-event text differs from instances.log: {raw_events_path}"
        )
    module.normalize_word_emissions(streamed, "de")
    return streamed.delays_ms, streamed.elapsed_ms


def main() -> None:
    args = parse_args()
    module = load_live_eval()
    config = json.loads((args.run_dir / "run_config.json").read_text(encoding="utf-8"))
    if str(config.get("target_lang")) != "de":
        raise ValueError(f"word-emission repair is only valid for target_lang=de: {args.run_dir}")
    provider = args.provider or str(config.get("provider") or "")
    if provider not in {"openai", "gemini"}:
        raise ValueError(f"unsupported provider: {provider}")

    instances_path = args.run_dir / "instances.log"
    responses_path = args.run_dir / "responses.jsonl"
    instances = read_jsonl(instances_path)
    responses = read_jsonl(responses_path)
    responses_by_index = {int(row["index"]): row for row in responses}
    repaired = 0
    for row in instances:
        index = int(row["index"])
        source = str((row.get("source") or [""])[0])
        sample_dir = find_sample_dir(args.run_dir, index, source)
        delays, elapsed = repaired_emissions(
            module,
            sample_dir / "raw_events.jsonl",
            provider,
            str(row.get("prediction") or ""),
        )
        row["delays"] = delays
        row["elapsed"] = elapsed
        row["prediction_length"] = len(delays)
        if index in responses_by_index:
            responses_by_index[index]["prediction_units"] = len(delays)
        repaired += 1

    for path in [instances_path, responses_path]:
        backup = path.with_name(f"{path.name}.pre_word_emission_fix")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
    atomic_write_jsonl(instances_path, instances)
    atomic_write_jsonl(responses_path, responses)
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "provider": provider,
                "repaired_rows": repaired,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
