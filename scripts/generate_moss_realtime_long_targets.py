#!/usr/bin/env python3
"""Generate row-level long MOSS-TTS-Realtime target speech (v2 pipeline).

Each input row carries the full Chinese target text of one InfiniSST row,
pre-split into generation groups. Every group is synthesized with the same
fixed reference voice, sanity-checked against a duration budget to catch
runaway generation, retried on failure, and finally concatenated into one
row wav.
"""
from __future__ import annotations

import argparse
import io
import json
import time
import unicodedata
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--rejected-jsonl", required=True)
    parser.add_argument("--wav-dir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="OpenMOSS-Team/MOSS-TTS-Realtime")
    parser.add_argument("--fixed-ref", default=None, help="Fixed reference wav path (server-visible). Omit for the model's unconditioned voice.")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-seconds-per-char", type=float, default=0.6)
    parser.add_argument("--min-runaway-floor-s", type=float, default=15.0)
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
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def existing_ids(path: Path, key: str) -> set[str]:
    if not path.exists():
        return set()
    return {str(row[key]) for row in read_jsonl(path) if row.get(key)}


def spoken_chars(text: str) -> int:
    return sum(1 for ch in text if unicodedata.category(ch)[0] in {"L", "N"})


def post_speech(url: str, payload: dict[str, Any], timeout_s: float) -> bytes:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"http_{exc.code}: {body[:500]}") from exc


def parse_wav(data: bytes) -> tuple[int, int, bytes]:
    """Return (sample_rate, num_frames, pcm_frames) for a mono 16-bit wav."""
    with wave.open(io.BytesIO(data), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(
                f"expected mono 16-bit wav, got ch={handle.getnchannels()} width={handle.getsampwidth()}"
            )
        rate = handle.getframerate()
        frames = handle.getnframes()
        return rate, frames, handle.readframes(frames)


def main() -> None:
    args = parse_args()
    output_jsonl = Path(args.output_jsonl)
    rejected_jsonl = Path(args.rejected_jsonl)
    wav_dir = Path(args.wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)

    done = existing_ids(output_jsonl, "row_id") | existing_ids(rejected_jsonl, "row_id")
    rows = [row for row in read_jsonl(args.input_jsonl) if str(row["row_id"]) not in done]

    url = f"{args.base_url.rstrip('/')}/v1/audio/speech"
    accepted = rejected = 0
    for processed, row in enumerate(rows, 1):
        row_id = str(row["row_id"])
        segments = row["segments"]
        groups = row["groups"]
        group_results: list[dict[str, Any]] = []
        failure: str | None = None
        pcm_parts: list[bytes] = []
        sample_rate: int | None = None
        for group_idx, group in enumerate(groups):
            text = "".join(segments[i]["text"] for i in group)
            budget_s = max(args.min_runaway_floor_s, spoken_chars(text) * args.max_seconds_per_char)
            attempt_durations: list[float] = []
            group_ok = False
            for _attempt in range(args.retries + 1):
                payload: dict[str, Any] = {
                    "model": args.model,
                    "voice": "default",
                    "input": text,
                    "response_format": "wav",
                }
                if args.fixed_ref:
                    payload["ref_audio"] = args.fixed_ref
                started = time.perf_counter()
                try:
                    data = post_speech(url, payload, args.timeout_s)
                    rate, frames, pcm = parse_wav(data)
                except Exception as exc:  # noqa: BLE001
                    failure = f"group{group_idx}: {exc}"
                    attempt_durations.append(-1.0)
                    continue
                duration = frames / rate
                attempt_durations.append(round(duration, 3))
                if duration > budget_s:
                    failure = (
                        f"group{group_idx}: runaway duration {duration:.1f}s > budget {budget_s:.1f}s"
                    )
                    continue
                if sample_rate is None:
                    sample_rate = rate
                elif rate != sample_rate:
                    failure = f"group{group_idx}: sample rate {rate} != {sample_rate}"
                    break
                pcm_parts.append(pcm)
                group_results.append(
                    {
                        "group": group_idx,
                        "chars": len(text),
                        "duration_s": round(duration, 3),
                        "attempt_durations": attempt_durations,
                        "request_s": round(time.perf_counter() - started, 3),
                    }
                )
                failure = None
                group_ok = True
                break
            if not group_ok:
                break

        if failure is not None or sample_rate is None:
            rejected += 1
            append_jsonl(
                rejected_jsonl,
                {"row_id": row_id, "split": row.get("split"), "error": failure, "groups": group_results},
            )
        else:
            wav_path = wav_dir / f"{row_id}.wav"
            with wave.open(str(wav_path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(sample_rate)
                for pcm in pcm_parts:
                    out.writeframes(pcm)
            total_s = sum(g["duration_s"] for g in group_results)
            accepted += 1
            append_jsonl(
                output_jsonl,
                {
                    "row_id": row_id,
                    "split": row.get("split"),
                    "wav": str(wav_path),
                    "sample_rate": sample_rate,
                    "duration_s": round(total_s, 3),
                    "spoken_chars": row.get("spoken_chars"),
                    "num_segments": len(segments),
                    "groups": group_results,
                    "fixed_ref": args.fixed_ref,
                },
            )
        if args.log_every > 0 and processed % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "processed": processed,
                        "accepted": accepted,
                        "rejected": rejected,
                        "remaining": len(rows) - processed,
                    }
                ),
                flush=True,
            )

    print(
        json.dumps(
            {
                "input": args.input_jsonl,
                "selected": len(rows),
                "accepted": accepted,
                "rejected": rejected,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
