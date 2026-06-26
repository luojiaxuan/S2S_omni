#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.floras_live import env_api_key
from s2s_omni.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe live S2S generated wavs with OpenAI ASR.")
    parser.add_argument("--run-output-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.environ.get("OPENAI_ASR_MODEL", "gpt-4o-mini-transcribe"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
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


def transcode_for_upload(input_path: Path, tmp_dir: Path, max_upload_bytes: int) -> Path:
    if input_path.stat().st_size <= max_upload_bytes:
        return input_path
    output_path = tmp_dir / f"{input_path.stem}.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "64k",
            str(output_path),
        ],
        check=True,
    )
    if output_path.stat().st_size > max_upload_bytes:
        raise RuntimeError(f"transcoded audio is still too large: {output_path.stat().st_size} bytes")
    return output_path


def multipart_body(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = "----s2s-omni-openai-asr-boundary"
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode(),
            b"Content-Type: application/octet-stream\r\n\r\n",
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), boundary


def transcribe(api_key: str, base_url: str, model: str, audio_path: Path) -> str:
    body, boundary = multipart_body(
        {"model": model, "response_format": "json"},
        "file",
        audio_path,
    )
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/audio/transcriptions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"transcription failed: HTTP {exc.code}: {body_text}") from exc
    return str(data.get("text") or "").strip()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    done = existing_rows(output_path) if args.resume else {}
    rows = []
    api_key = env_api_key()
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
            text = transcribe(api_key, args.base_url, args.model, upload_path)
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
