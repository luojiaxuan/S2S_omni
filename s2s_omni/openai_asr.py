from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


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


def transcribe_openai(api_key: str, base_url: str, model: str, audio_path: Path) -> str:
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
