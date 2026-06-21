#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_samples, write_jsonl
from s2s_omni.prompts import SYSTEM_COMPRESSION, build_compression_user_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-Omni baseline through chat API.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def chat(args: argparse.Namespace, messages: list[dict[str, str]], audio_path: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "modalities": ["text"] if args.text_only else ["text", "audio"],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if audio_path:
        body["audios"] = [audio_path]
    if not args.text_only:
        body["audio"] = {"format": "wav"}
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def main() -> None:
    args = parse_args()
    records = []
    for sample in read_samples(args.input):
        messages = [
            {"role": "system", "content": SYSTEM_COMPRESSION.strip()},
            {"role": "user", "content": build_compression_user_prompt(sample, include_reference=False)},
        ]
        response = chat(args, messages, sample.audio_path)
        choice = response["choices"][0]
        content = choice.get("message", {}).get("content") or ""
        records.append(
            {
                "id": sample.id,
                "candidate_translation": content,
                "raw_response": response,
            }
        )
        print(f"{sample.id}: {content[:120]}")
    write_jsonl(args.output, records)
    print(f"wrote {len(records)} predictions to {Path(args.output)}")


if __name__ == "__main__":
    main()
