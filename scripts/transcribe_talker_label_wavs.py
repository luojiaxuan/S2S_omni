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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe generated talker-label wavs for coverage audit.")
    parser.add_argument("--input", required=True, help="talker_code_labels.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--wav-root", help="Optional local directory for wav basenames.")
    parser.add_argument("--model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def resolve_wav_path(record: dict[str, Any], input_path: Path, wav_root: str | None) -> Path | None:
    raw = record.get("wav_path") or record.get("evo_audio")
    candidates: list[Path] = []
    if raw:
        path = Path(str(raw))
        candidates.append(path)
        if not path.is_absolute():
            candidates.append(input_path.parent / path)
        if wav_root:
            candidates.append(Path(wav_root) / path.name)
    if wav_root and raw:
        candidates.append(Path(wav_root) / Path(str(raw)).name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def make_pipeline(args: argparse.Namespace):
    import torch
    from transformers import pipeline

    device = args.device
    torch_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    return pipeline(
        "automatic-speech-recognition",
        model=args.model,
        torch_dtype=torch_dtype,
        device=device,
    )


def transcribe(pipe: Any, wav_path: Path, language: str) -> str:
    result = pipe(
        str(wav_path),
        generate_kwargs={"language": language, "task": "transcribe"},
        return_timestamps=False,
    )
    if isinstance(result, dict):
        return str(result.get("text") or "").strip()
    return str(result).strip()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    records = read_jsonl(input_path)
    if args.max_records > 0:
        records = records[: args.max_records]
    pipe = make_pipeline(args)
    rows = []
    for index, record in enumerate(records, start=1):
        wav_path = resolve_wav_path(record, input_path, args.wav_root)
        row = {
            "id": record.get("id"),
            "wav_path": str(wav_path) if wav_path else None,
            "asr_model": args.model,
            "asr_language": args.language,
        }
        if wav_path is None:
            row["error"] = "wav_not_found"
            row["asr_text"] = ""
        else:
            try:
                row["asr_text"] = transcribe(pipe, wav_path, args.language)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["asr_text"] = ""
        rows.append(row)
        if args.log_every > 0 and index % args.log_every == 0:
            print(json.dumps({"processed": index, "last_id": row["id"]}, ensure_ascii=False), flush=True)
    write_jsonl(args.output, rows)
    print(json.dumps({"records": len(rows), "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
