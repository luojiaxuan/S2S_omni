#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a KIT Lecture Translator session.")
    parser.add_argument("--cookie-header-file", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--base-url", default="https://lt2srv.iar.kit.edu")
    parser.add_argument("--name", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--mt-language", default="zh")
    parser.add_argument("--audio-language", default="zh")
    parser.add_argument("--tts-quality-mode", default="high_quality")
    parser.add_argument("--format", default="mixed", choices=["online", "mixed"])
    parser.add_argument("--availability", default="private")
    parser.add_argument("--smart-chaptering", default="online_dynamic")
    parser.add_argument("--logging", default="1")
    parser.add_argument("--profanity", default="1")
    parser.add_argument("--filter-music", default="1")
    parser.add_argument("--type", default="lecture")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_cookie_header(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def create_url(args: argparse.Namespace) -> str:
    params = {
        "name": args.name,
        "legals": "1",
        "language": args.language,
        "mtLanguage": args.mt_language,
        "audioLanguage": args.audio_language,
        "ttsQualityMode": args.tts_quality_mode,
        "format": args.format,
        "availability": args.availability,
        "smartChaptering": args.smart_chaptering,
        "logging": args.logging,
        "profanity": args.profanity,
        "filter_music": args.filter_music,
        "type": args.type,
    }
    return f"{args.base_url.rstrip('/')}/create?{urllib.parse.urlencode(params)}"


def session_id_from_url(url: str) -> str:
    match = re.search(r"/present/([0-9]+)", url)
    return match.group(1) if match else ""


def main() -> None:
    args = parse_args()
    cookie = read_cookie_header(Path(args.cookie_header_file).expanduser())
    url = create_url(args)
    request = urllib.request.Request(
        url,
        headers={"Cookie": cookie, "User-Agent": "Mozilla/5.0"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=60.0) as response:
        body = response.read(2000).decode("utf-8", errors="replace")
        final_url = response.geturl()
        payload: dict[str, Any] = {
            "created_at": now_iso(),
            "ok": 200 <= int(response.status) < 300,
            "status": int(response.status),
            "name": args.name,
            "url": url,
            "final_url": final_url,
            "session_id": session_id_from_url(final_url),
            "body_start": body,
            "config": {
                "language": args.language,
                "mtLanguage": args.mt_language,
                "audioLanguage": args.audio_language,
                "ttsQualityMode": args.tts_quality_mode,
                "format": args.format,
                "availability": args.availability,
                "smartChaptering": args.smart_chaptering,
            },
        }
    if not payload["session_id"]:
        raise SystemExit(f"could not parse session id from {payload['final_url']}")
    output = Path(args.output_json).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"session_id": payload["session_id"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
