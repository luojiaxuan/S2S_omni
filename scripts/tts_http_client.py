#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI-compatible /v1/audio/speech client.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/audio/speech")
    parser.add_argument("--text", default=None, help="Defaults to S2S_TEXT.")
    parser.add_argument("--output", default=None, help="Defaults to S2S_OUTPUT_WAV.")
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--token-count", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    return parser.parse_args()


def maybe_set(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is not None and value != "":
        payload[key] = value


def env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def main() -> None:
    args = parse_args()
    text = args.text if args.text is not None else os.environ.get("S2S_TEXT", "")
    output = Path(args.output or os.environ.get("S2S_OUTPUT_WAV", "output.wav"))
    ref_audio = args.ref_audio or os.environ.get("S2S_REF_AUDIO") or os.environ.get("S2S_SOURCE_AUDIO")
    ref_text = args.ref_text or os.environ.get("S2S_REF_TEXT") or os.environ.get("S2S_SOURCE_TEXT")
    if not text.strip():
        raise ValueError("empty --text/S2S_TEXT")

    payload: dict[str, Any] = {"input": text}
    maybe_set(payload, "language", args.language)
    maybe_set(payload, "temperature", args.temperature)
    maybe_set(payload, "top_k", args.top_k)
    maybe_set(payload, "top_p", args.top_p)
    maybe_set(payload, "max_new_tokens", args.max_new_tokens or env_int("S2S_TTS_MAX_NEW_TOKENS"))
    maybe_set(payload, "token_count", args.token_count or env_int("S2S_TTS_TOKEN_COUNT"))
    if ref_audio:
        payload["references"] = [{"audio_path": ref_audio, "text": ref_text or ""}]

    response = requests.post(args.url, json=payload, timeout=args.timeout_s)
    response.raise_for_status()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response.content)
    print({"output": str(output), "bytes": len(response.content)}, flush=True)


if __name__ == "__main__":
    main()
