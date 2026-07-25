from __future__ import annotations

import fcntl
import json
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ASR_REQUEST_SLOTS = 4
ASR_SLOT_DIR = Path("/tmp/s2s_omni_openai_asr_slots")


@contextmanager
def transcription_request_slot() -> Iterator[None]:
    ASR_SLOT_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        for index in range(ASR_REQUEST_SLOTS):
            handle = (ASR_SLOT_DIR / f"{index}.lock").open("a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
            return
        time.sleep(0.1)


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
        raise RuntimeError(
            f"transcoded audio is still too large: {output_path.stat().st_size} bytes"
        )
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


def transcribe_openai_json(
    api_key: str,
    base_url: str,
    model: str,
    audio_path: Path,
    *,
    response_format: str = "json",
    language: str = "",
    timestamp_granularities: tuple[str, ...] = (),
    max_attempts: int = 5,
    retry_base_s: float = 1.0,
) -> dict[str, Any]:
    fields = {"model": model, "response_format": response_format}
    if language:
        fields["language"] = language
    if timestamp_granularities:
        fields["timestamp_granularities[]"] = ",".join(timestamp_granularities)
    body, boundary = multipart_body(
        fields,
        "file",
        audio_path,
    )
    retryable_statuses = {408, 409, 429, 500, 502, 503, 504}
    data: Any = None
    for attempt in range(1, max_attempts + 1):
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
            with (
                transcription_request_slot(),
                urllib.request.urlopen(request, timeout=180.0) as resp,
            ):
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            if exc.code not in retryable_statuses or attempt == max_attempts:
                raise RuntimeError(
                    f"transcription failed: HTTP {exc.code}: {body_text}"
                ) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt == max_attempts:
                raise RuntimeError(f"transcription failed after {max_attempts} attempts") from exc
        time.sleep(retry_base_s * (2 ** (attempt - 1)))
    if not isinstance(data, dict):
        raise TypeError(f"unexpected transcription response: {type(data).__name__}")
    return data


def transcribe_openai(api_key: str, base_url: str, model: str, audio_path: Path) -> str:
    data = transcribe_openai_json(api_key, base_url, model, audio_path)
    return str(data.get("text") or "").strip()
