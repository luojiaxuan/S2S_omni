#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a KIT raw run into a FLORAS live result.json.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-json", required=True)
    parser.add_argument("--target-wav", required=True)
    parser.add_argument("--audio-chunks-jsonl", required=True)
    parser.add_argument("--source-eval-wav", required=True)
    parser.add_argument("--source-stream-wav", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--chunk-ms", type=int, default=1920)
    parser.add_argument("--backend", default="kit_lecture_translator")
    parser.add_argument("--model", default="kit_mixed_high_quality")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def manifest_row(path: Path, run_id: str) -> dict[str, Any]:
    matches = [row for row in read_jsonl(path) if str(row.get("run_id") or "") == run_id]
    if len(matches) != 1:
        raise SystemExit(f"expected one {run_id} row in {path}, found {len(matches)}")
    return matches[0]


def stable_text(messages: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    for msg in sorted(messages, key=lambda row: int(row.get("message_id") or 0)):
        text = str(msg.get("text") or "").strip()
        if not text or text == "TTS-finish" or msg.get("unstable") is True:
            continue
        pieces.append(text)
    return "".join(pieces)


def validate_audio_chunks(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise SystemExit(f"expected at least one audio chunk row in {path}")
    required = {"arrival_s", "audio_duration_s", "cumulative_audio_duration_s"}
    for index, row in enumerate(rows):
        missing = sorted(required.difference(row))
        if missing:
            raise SystemExit(f"{path} row {index} is missing required timing fields: {missing}")
        for key in required:
            try:
                float(row[key])
            except (TypeError, ValueError):
                raise SystemExit(f"{path} row {index} has non-numeric {key}: {row[key]!r}") from None
    return rows


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest).expanduser()
    run_json = Path(args.run_json).expanduser()
    target_wav = Path(args.target_wav).expanduser()
    source_eval_wav = Path(args.source_eval_wav).expanduser()
    source_stream_wav = Path(args.source_stream_wav).expanduser()
    audio_chunks = Path(args.audio_chunks_jsonl).expanduser()
    run = json.loads(run_json.read_text(encoding="utf-8"))
    row = manifest_row(manifest, args.run_id)
    tts_messages = run.get("collection", {}).get("messagesByComponent", {}).get("tts:0", [])
    audio_chunk_rows = validate_audio_chunks(audio_chunks)
    result = {
        "run_id": args.run_id,
        "direction": row.get("direction"),
        "backend": args.backend,
        "model": args.model,
        "chunk_ms": args.chunk_ms,
        "speed_factor": float(row.get("speed_factor") or 1.0),
        "source_eval_duration_s": round(wav_duration_s(source_eval_wav), 6),
        "source_stream_duration_s": round(wav_duration_s(source_stream_wav), 6),
        "generated_duration_s": round(wav_duration_s(target_wav), 6),
        "source_eval_wav_path": str(source_eval_wav),
        "source_stream_wav_path": str(source_stream_wav),
        "generated_wav_path": str(target_wav),
        "audio_chunks_path": str(audio_chunks),
        "raw_events_path": str(run_json),
        "target_language": row.get("target_lang"),
        "target_lang": row.get("target_lang"),
        "target_reference_text": row.get("target_reference_text"),
        "input_transcript": row.get("source_transcript"),
        "output_transcript": stable_text(tts_messages),
        "session_name": run.get("sessionName"),
        "kit_config": run.get("config"),
        "candidate_text_source": "target_speech_asr_gpt4o_mini_transcribe",
        "audio_chunk_count": len(audio_chunk_rows),
    }
    output = Path(args.output_json).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "duration_s": result["generated_duration_s"]}, indent=2))


if __name__ == "__main__":
    main()
