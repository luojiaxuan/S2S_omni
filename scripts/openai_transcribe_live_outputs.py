#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import write_jsonl
from s2s_omni.openai_asr import transcode_for_upload, transcribe_openai


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe live S2S generated wavs with OpenAI ASR.")
    parser.add_argument("--run-output-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-4o-mini-transcribe")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--max-upload-mb", type=float, default=24.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_results(run_output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(run_output_dir.glob("*/result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["result_path"] = str(path)
        rows.append(row)
    return rows


def existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("run_id"):
            out[str(row["run_id"])] = row
    return out


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    done = existing_rows(output_path) if args.resume else {}
    rows = []
    max_upload_bytes = int(args.max_upload_mb * 1024 * 1024)
    with tempfile.TemporaryDirectory(prefix="s2s_omni_asr_") as tmp:
        tmp_dir = Path(tmp)
        for result in load_results(Path(args.run_output_dir)):
            run_id = str(result["run_id"])
            if run_id in done:
                rows.append(done[run_id])
                continue
            wav_path = Path(str(result["generated_wav_path"]))
            upload_path = transcode_for_upload(wav_path, tmp_dir, max_upload_bytes)
            text = transcribe_openai(args.api_key, args.base_url, args.model, upload_path)
            row = {
                "run_id": run_id,
                "asr_text": text,
                "asr_model": args.model,
                "source_audio_path": str(wav_path),
                "uploaded_audio_path": str(upload_path if upload_path == wav_path else upload_path.name),
                "uploaded_audio_size_bytes": upload_path.stat().st_size,
            }
            rows.append(row)
            write_jsonl(output_path, rows)
            print(json.dumps({"run_id": run_id, "asr_chars": len(text)}, ensure_ascii=False), flush=True)
    write_jsonl(output_path, rows)
    print(json.dumps({"rows": len(rows), "output": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
