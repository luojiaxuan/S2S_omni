#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import unicodedata
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MOSS-TTS-Realtime target wavs and write MOSS finetuning raw JSONL."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--rejected-jsonl", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="OpenMOSS-Team/MOSS-TTS-Realtime")
    parser.add_argument("--voice", default="default")
    parser.add_argument("--response-format", default="wav")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--path-mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--include-user-turn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-rejections", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
        handle.flush()


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for row in read_jsonl(path):
        if row.get("id"):
            ids.add(str(row["id"]))
    return ids


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def emit_path(root: Path, path: Path, mode: str) -> str:
    if mode == "absolute":
        return str(path.resolve())
    return str(path.relative_to(root))


def audio_duration_s(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            return float(frames) / float(rate) if rate else None
    except Exception:
        return None


def has_spoken_content(text: str) -> bool:
    return any(unicodedata.category(ch)[0] in {"L", "N"} for ch in text)


def post_json_bytes(url: str, payload: dict[str, Any], timeout_s: float) -> bytes:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"http_{exc.code}: {body[:500]}") from exc


def build_raw_record(
    row: dict[str, Any],
    root: Path,
    ref_wav: Path,
    target_wav: Path,
    path_mode: str,
    include_user_turn: bool,
) -> dict[str, Any]:
    conversations: list[dict[str, Any]] = []
    if include_user_turn:
        conversations.append(
            {
                "role": "user",
                "text": "",
                "wav": emit_path(root, ref_wav, path_mode),
            }
        )
    conversations.append(
        {
            "role": "assistant",
            "text": str(row["target_text"]),
            "wav": emit_path(root, target_wav, path_mode),
        }
    )
    return {
        "id": str(row["id"]),
        "ref_wav": emit_path(root, ref_wav, path_mode),
        "conversations": conversations,
        "metadata": {
            "split": row.get("split"),
            "src_lang": "en",
            "tgt_lang": "zh",
            "target_duration_s": audio_duration_s(target_wav),
            "source_duration_s": audio_duration_s(ref_wav),
            **dict(row.get("metadata") or {}),
        },
    }


def generate_one(row: dict[str, Any], args: argparse.Namespace, root: Path) -> tuple[bool, dict[str, Any]]:
    sample_id = str(row["id"])
    ref_wav = resolve_path(root, str(row["ref_wav"]))
    output_wav = resolve_path(root, str(row["output_wav"]))
    try:
        if not ref_wav.exists():
            raise FileNotFoundError(f"missing ref_wav: {ref_wav}")
        target_text = str(row.get("target_text") or row.get("input") or "").strip()
        if not target_text:
            raise ValueError("empty target_text")
        if not has_spoken_content(target_text):
            raise ValueError("no spoken target_text")
        if not output_wav.exists() or output_wav.stat().st_size == 0:
            output_wav.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model": args.model,
                "voice": args.voice,
                "input": target_text,
                "ref_audio": str(ref_wav),
                "response_format": args.response_format,
            }
            started = time.perf_counter()
            wav_bytes = post_json_bytes(
                f"{args.base_url.rstrip('/')}/v1/audio/speech",
                payload,
                args.timeout_s,
            )
            elapsed = time.perf_counter() - started
            output_wav.write_bytes(wav_bytes)
        else:
            elapsed = 0.0
        raw = build_raw_record(
            row=row,
            root=root,
            ref_wav=ref_wav,
            target_wav=output_wav,
            path_mode=args.path_mode,
            include_user_turn=args.include_user_turn,
        )
        raw["metadata"]["moss_realtime_base_url"] = args.base_url
        raw["metadata"]["moss_realtime_model"] = args.model
        raw["metadata"]["tts_request_s"] = round(elapsed, 6)
        return True, raw
    except Exception as exc:
        rejected = {
            "id": sample_id,
            "accepted": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "row": row,
        }
        return False, rejected


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    output_jsonl = Path(args.output_jsonl)
    rejected_jsonl = Path(args.rejected_jsonl)
    if not args.resume:
        for path in (output_jsonl, rejected_jsonl):
            if path.exists():
                path.unlink()
    done = existing_ids(output_jsonl) | existing_ids(rejected_jsonl)
    rows = [row for row in read_jsonl(args.input_jsonl) if str(row.get("id")) not in done]

    accepted = rejected = processed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_row = {
            executor.submit(generate_one, row, args, root): row
            for row in rows
        }
        for future in concurrent.futures.as_completed(future_to_row):
            ok, payload = future.result()
            processed += 1
            if ok:
                accepted += 1
                append_jsonl(output_jsonl, payload)
            else:
                rejected += 1
                append_jsonl(rejected_jsonl, payload)
            if args.log_every > 0 and processed % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "accepted": accepted,
                            "rejected": rejected,
                            "remaining": len(rows) - processed,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    print(
        json.dumps(
            {
                "input": args.input_jsonl,
                "output_jsonl": str(output_jsonl),
                "rejected_jsonl": str(rejected_jsonl),
                "selected": len(rows),
                "accepted": accepted,
                "rejected": rejected,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if rejected and not args.allow_rejections:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
