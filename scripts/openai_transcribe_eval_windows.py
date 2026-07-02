#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.floras_live import env_api_key
from s2s_omni.io import write_jsonl
from s2s_omni.openai_asr import transcode_for_upload, transcribe_openai


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe per-window target wavs from FLORAS eval output.")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.environ.get("OPENAI_ASR_MODEL", "gpt-4o-mini-transcribe"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--max-upload-mb", type=float, default=24.0)
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--window-index", type=int, action="append", default=[])
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def existing_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("run_id") is not None and row.get("window_index") is not None:
            out[(str(row["run_id"]), int(row["window_index"]))] = row
    return out


def load_windows(
    eval_dir: Path,
    *,
    run_ids: set[str],
    window_indexes: set[int],
    max_windows: int,
) -> list[dict[str, Any]]:
    windows = []
    for path in sorted(eval_dir.glob("*/timeline.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            run_id = str(row.get("run_id") or "")
            window_index = int(row.get("window_index", -1))
            if run_ids and run_id not in run_ids:
                continue
            if window_indexes and window_index not in window_indexes:
                continue
            audio_path = Path(str(row.get("target_window_audio_path") or ""))
            if not audio_path.exists():
                audio_path = path.parent / str(row.get("target_window_audio_rel") or "")
            if not audio_path.exists():
                continue
            row["target_window_audio_path"] = str(audio_path)
            windows.append(row)
            if max_windows > 0 and len(windows) >= max_windows:
                return windows
    return windows


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    done = existing_rows(output_path) if args.resume else {}
    rows = []
    api_key = env_api_key()
    max_upload_bytes = int(args.max_upload_mb * 1024 * 1024)
    with tempfile.TemporaryDirectory(prefix="s2s_omni_window_asr_") as tmp:
        tmp_dir = Path(tmp)
        for window in load_windows(
            Path(args.eval_dir),
            run_ids=set(args.run_id),
            window_indexes=set(args.window_index),
            max_windows=args.max_windows,
        ):
            run_id = str(window["run_id"])
            window_index = int(window["window_index"])
            key = (run_id, window_index)
            if key in done:
                rows.append(done[key])
                continue
            wav_path = Path(str(window["target_window_audio_path"]))
            upload_path = transcode_for_upload(wav_path, tmp_dir, max_upload_bytes)
            text = transcribe_openai(api_key, args.base_url, args.model, upload_path)
            row = {
                "run_id": run_id,
                "window_index": window_index,
                "asr_text": text,
                "asr_model": args.model,
                "target_window_audio_path": str(wav_path),
                "target_window_duration_s": window.get("target_window_duration_s"),
                "uploaded_audio_size_bytes": upload_path.stat().st_size,
            }
            rows.append(row)
            write_jsonl(output_path, rows)
            print(
                json.dumps(
                    {
                        "run_id": run_id,
                        "window_index": window_index,
                        "asr_chars": len(text),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    write_jsonl(output_path, rows)
    print(json.dumps({"rows": len(rows), "output": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
