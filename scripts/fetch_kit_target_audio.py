#!/usr/bin/env python3
"""Resolve KIT Lecture Translator tts:0 linked-audio chunks into a single WAV.

KIT delivers synthesized target speech as *linked* data: each tts:0 message
carries ``b64_enc_pcm_s16le`` (or ``b64_enc_opus``) whose value is a reference
such as ``Data/{session}/{N}`` rather than inline base64. The web client turns
the live ``/ltapi/...`` form into ``/webapi/...`` and fetches base64 PCM which it
decodes as signed 16-bit little-endian mono at 16 kHz.

This script reads a run JSON produced by ``run_kit_live_floras.py`` (which stores
the tts:0 messages under ``collection.messagesByComponent``), resolves every
audio chunk, concatenates them in message order, and writes ``target.wav``.
Because the exact live URL shape is not documented, the first chunk is probed
against a list of candidate URL templates and the working template is reused.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

BASE_URL = "https://lecture-translator.kit.edu"
SAMPLE_RATE = 16000  # KIT client decodes tts PCM as 16 kHz mono s16le


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve KIT tts:0 linked audio into a WAV.")
    parser.add_argument("--run-json", required=True, help="run_kit_live_floras.py output JSON.")
    parser.add_argument("--cookie-header-file", required=True)
    parser.add_argument("--output-wav", required=True)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--component", default="tts:0")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    return parser.parse_args()


def read_cookie_header(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def http_get(url: str, cookie_header: str, timeout_s: float) -> tuple[int, str, bytes]:
    request = urllib.request.Request(
        url,
        headers={"Cookie": cookie_header, "User-Agent": "Mozilla/5.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return int(response.status), response.headers.get_content_type(), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), "error", exc.read()


def candidate_urls(base_url: str, session_id: str, link: str) -> list[str]:
    """Return candidate absolute URLs for a stored audio link.

    Observed stored form is ``Data/{session}/{N}``; the live form embeds
    ``/ltapi/...`` which the client rewrites to ``/webapi/...``.
    """
    link = link.strip().strip('"')
    tail = link
    if "/ltapi" in link:
        tail = link[link.index("/ltapi") + len("/ltapi"):].lstrip("/")
    elif link.startswith("/webapi/"):
        tail = link[len("/webapi/"):]
    tail = tail.lstrip("/")
    # ``tail`` is expected to look like ``Data/{session}/{N}`` or ``{session}/{N}``.
    suffix = tail[len("Data/"):] if tail.startswith("Data/") else tail
    return [
        f"{base_url}/webapi/{tail}",
        f"{base_url}/webapi/Data/{suffix}",
        f"{base_url}/webapi/{session_id}/getData/{suffix.split('/')[-1]}",
        f"{base_url}/webapi/getData/{tail}",
        f"{base_url}/ltapi/{tail}",
        f"{base_url}/{tail}",
    ]


def decode_pcm(body: bytes) -> bytes | None:
    """Interpret a fetched chunk body as PCM s16le bytes.

    The client fetches text and ``atob``s it, so the body is normally base64
    text. Fall back to treating the body as raw PCM if it is not base64.
    """
    text = body.strip()
    if not text:
        return None
    if text[:1] in (b"<", b"{"):
        return None  # HTML/JSON error, not audio
    try:
        decoded = base64.b64decode(text, validate=True)
        if decoded and len(decoded) % 2 == 0:
            return decoded
    except (binascii.Error, ValueError):
        pass
    if len(body) % 2 == 0 and len(body) > 0:
        return body
    return None


def load_tts_messages(run: dict[str, Any], component: str) -> list[dict[str, Any]]:
    components = run.get("collection", {}).get("messagesByComponent", {})
    messages = components.get(component, [])
    audio_msgs = [
        m
        for m in messages
        if isinstance(m, dict) and (m.get("b64_enc_pcm_s16le") or m.get("b64_enc_opus"))
    ]
    return sorted(audio_msgs, key=lambda m: int(m.get("message_id") or 0))


def main() -> None:
    args = parse_args()
    run = json.loads(Path(args.run_json).expanduser().read_text(encoding="utf-8"))
    cookie_header = read_cookie_header(Path(args.cookie_header_file).expanduser())
    session_id = str(run.get("sessionId") or "")
    messages = load_tts_messages(run, args.component)
    if not messages:
        raise SystemExit(f"No {args.component} audio messages in {args.run_json}")
    if any(m.get("b64_enc_opus") for m in messages):
        print("WARNING: b64_enc_opus chunks present; this resolver handles PCM only.")

    working_template: str | None = None
    pcm_parts: list[bytes] = []
    resolved = 0
    for index, msg in enumerate(messages):
        link = str(msg.get("b64_enc_pcm_s16le") or "")
        if not link:
            continue
        # Inline base64 (no path separators) is used directly.
        inline = decode_pcm(link.encode("ascii", "ignore")) if "/" not in link else None
        if inline is not None:
            pcm_parts.append(inline)
            resolved += 1
            continue
        urls = (
            [working_template.format(link=link, session=session_id)]
            if working_template
            else candidate_urls(args.base_url, session_id, link)
        )
        chunk_pcm: bytes | None = None
        for url in urls:
            status, ctype, body = http_get(url, cookie_header, args.timeout_s)
            if status == 200:
                chunk_pcm = decode_pcm(body)
                if chunk_pcm is not None:
                    if working_template is None:
                        print(f"Locked audio URL shape via: {url}")
                    chunk_pcm = chunk_pcm
                    break
        if chunk_pcm is None:
            print(f"  chunk {index} ({link}) unresolved")
            continue
        pcm_parts.append(chunk_pcm)
        resolved += 1

    if not pcm_parts:
        raise SystemExit("No audio chunks resolved; the session data may have expired.")

    pcm = b"".join(pcm_parts)
    out_path = Path(args.output_wav).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    duration_s = len(pcm) / 2 / SAMPLE_RATE
    print(
        json.dumps(
            {
                "output_wav": str(out_path),
                "chunks_total": len(messages),
                "chunks_resolved": resolved,
                "target_duration_s": round(duration_s, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
