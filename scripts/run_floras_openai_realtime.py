#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.floras_live import (
    OPENAI_TRANSLATION_MODEL,
    OPENAI_TRANSLATION_WS,
    REALTIME_CHUNK_MS,
    REALTIME_SAMPLE_RATE,
    append_jsonl,
    env_api_key,
    ffmpeg_convert,
    pcm16_duration_s,
    pcm16_to_wav,
    read_run_manifest,
    sanitize_id,
    target_language_for_direction,
    wav_to_pcm16,
)
from s2s_omni.io import write_jsonl
from s2s_omni.rasst import audio_duration_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FLORAS live S2S through OpenAI Realtime.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--smoke-seconds", type=float, default=0.0)
    parser.add_argument("--chunk-ms", type=int, default=REALTIME_CHUNK_MS)
    parser.add_argument("--model", default=os.environ.get("OPENAI_REALTIME_TRANSLATE_MODEL", OPENAI_TRANSLATION_MODEL))
    parser.add_argument("--ws-url", default=os.environ.get("OPENAI_REALTIME_TRANSLATE_WS", OPENAI_TRANSLATION_WS))
    parser.add_argument("--voice", default=os.environ.get("OPENAI_REALTIME_VOICE", ""))
    parser.add_argument("--instructions", default="")
    parser.add_argument("--session-update-type", default="session.update")
    parser.add_argument("--skip-session-update", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--session-config-json", default="")
    parser.add_argument("--receive-timeout-s", type=float, default=30.0)
    parser.add_argument("--pace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def select_runs(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.run_id:
        rows = [row for row in rows if str(row.get("run_id")) == args.run_id]
        if not rows:
            raise SystemExit(f"run_id not found: {args.run_id}")
    if args.max_runs > 0:
        rows = rows[: args.max_runs]
    return rows


def run_dir(output_dir: Path, run: dict[str, Any]) -> Path:
    return output_dir / sanitize_id(str(run["run_id"]))


def prepare_audio(run: dict[str, Any], out_dir: Path, smoke_seconds: float) -> dict[str, Any]:
    speed = float(run["speed_factor"])
    source_wav = Path(str(run["source_wav_path"]))
    source_eval = out_dir / "source_eval.wav"
    source_stream = out_dir / "source_stream_24k.wav"
    duration = smoke_seconds if smoke_seconds > 0 else None
    ffmpeg_convert(
        source_wav,
        source_eval,
        sample_rate=16000,
        speed=1.0,
        duration_s=duration,
    )
    ffmpeg_convert(
        source_eval,
        source_stream,
        sample_rate=REALTIME_SAMPLE_RATE,
        speed=speed,
    )
    return {
        "source_eval_wav_path": str(source_eval),
        "source_stream_wav_path": str(source_stream),
        "source_eval_duration_s": round(audio_duration_s(source_eval), 6),
        "source_stream_duration_s": round(audio_duration_s(source_stream), 6),
    }


def session_update_payload(args: argparse.Namespace, target_language: str) -> dict[str, Any]:
    if args.session_config_json:
        payload = json.loads(Path(args.session_config_json).read_text(encoding="utf-8"))
        if "type" not in payload:
            payload["type"] = args.session_update_type
        return payload
    output_config: dict[str, Any] = {"language": target_language}
    if args.voice:
        output_config["voice"] = args.voice
    session: dict[str, Any] = {"audio": {"output": output_config}}
    if args.instructions:
        session["instructions"] = args.instructions
    return {
        "type": args.session_update_type,
        "session": session,
    }


def sanitize_event(event: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    event_type = str(event.get("type") or "")
    for key, value in event.items():
        if isinstance(value, str) and key in {"audio", "delta", "data"} and "audio" in event_type:
            out[key] = f"<base64:{len(value)} chars>"
        else:
            out[key] = value
    return out


def decode_audio_delta(event: dict[str, Any]) -> bytes:
    event_type = str(event.get("type") or "")
    if event_type != "session.output_audio.delta":
        return b""
    for key in ["audio", "delta", "data"]:
        value = event.get(key)
        if isinstance(value, str) and value:
            try:
                return base64.b64decode(value)
            except Exception:
                return b""
    return b""


def output_transcript_delta(event: dict[str, Any]) -> str:
    if str(event.get("type") or "") != "session.output_transcript.delta":
        return ""
    for key in ["delta", "text", "transcript"]:
        value = event.get(key)
        if isinstance(value, str):
            return value
    return ""


def input_transcript_delta(event: dict[str, Any]) -> str:
    if str(event.get("type") or "") != "session.input_transcript.delta":
        return ""
    value = event.get("delta")
    return value if isinstance(value, str) else ""


async def connect_ws(url: str, headers: list[tuple[str, str]]):
    import websockets

    try:
        return await websockets.connect(url, additional_headers=headers, max_size=None)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers, max_size=None)


async def run_live(run: dict[str, Any], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    raw_events = out_dir / "raw_events.jsonl"
    audio_chunks = out_dir / "audio_chunks.jsonl"
    send_events = out_dir / "send_events.jsonl"
    for path in [raw_events, audio_chunks, send_events]:
        if path.exists():
            path.unlink()

    target_language = target_language_for_direction(str(run["direction"]))
    api_key = env_api_key()
    url = f"{args.ws_url}?model={args.model}"
    safety_id = os.environ.get("OPENAI_SAFETY_IDENTIFIER", "s2s-omni-floras-live")
    headers = [("Authorization", f"Bearer {api_key}"), ("OpenAI-Safety-Identifier", safety_id)]

    ws = await connect_ws(url, headers)
    output_audio = bytearray()
    output_transcript_parts: list[str] = []
    input_transcript_parts: list[str] = []
    errors: list[dict[str, Any]] = []
    closed = asyncio.Event()
    start_time = time.monotonic()
    cumulative_audio_s = 0.0
    chunk_index = 0

    async def receiver() -> None:
        nonlocal cumulative_audio_s, chunk_index
        try:
            while not closed.is_set():
                message = await ws.recv()
                now_s = time.monotonic() - start_time
                event = json.loads(message)
                event_type = str(event.get("type") or "")
                safe = sanitize_event(event)
                safe["received_at_s"] = round(now_s, 6)
                append_jsonl(raw_events, safe)
                if "error" in event_type or event.get("error"):
                    errors.append(safe)
                delta = decode_audio_delta(event)
                if delta:
                    duration_s = pcm16_duration_s(delta)
                    cumulative_audio_s += duration_s
                    output_audio.extend(delta)
                    append_jsonl(
                        audio_chunks,
                        {
                            "chunk_index": chunk_index,
                            "arrival_s": round(now_s, 6),
                            "audio_duration_s": round(duration_s, 6),
                            "cumulative_audio_duration_s": round(cumulative_audio_s, 6),
                            "bytes": len(delta),
                            "event_type": event_type,
                        },
                    )
                    chunk_index += 1
                output_delta = output_transcript_delta(event)
                if output_delta:
                    output_transcript_parts.append(output_delta)
                input_delta = input_transcript_delta(event)
                if input_delta:
                    input_transcript_parts.append(input_delta)
                if event_type in {"session.closed", "translation_session.closed"}:
                    closed.set()
        except Exception as exc:
            append_jsonl(raw_events, {"receiver_error": f"{type(exc).__name__}: {exc}"})
            closed.set()

    receiver_task = asyncio.create_task(receiver())
    if not args.skip_session_update:
        await ws.send(json.dumps(session_update_payload(args, target_language)))
        append_jsonl(send_events, {"type": "session_update", "sent_at_s": 0.0})

    pcm = wav_to_pcm16(run["source_stream_wav_path"], REALTIME_SAMPLE_RATE)
    samples_per_chunk = int(round(REALTIME_SAMPLE_RATE * args.chunk_ms / 1000.0))
    bytes_per_chunk = samples_per_chunk * 2
    send_index = 0
    for start in range(0, len(pcm), bytes_per_chunk):
        payload = pcm[start : start + bytes_per_chunk]
        await ws.send(
            json.dumps(
                {
                    "type": "session.input_audio_buffer.append",
                    "audio": base64.b64encode(payload).decode("ascii"),
                }
            )
        )
        append_jsonl(
            send_events,
            {
                "send_index": send_index,
                "sent_at_s": round(time.monotonic() - start_time, 6),
                "audio_duration_s": round(pcm16_duration_s(payload), 6),
                "bytes": len(payload),
            },
        )
        send_index += 1
        if args.pace:
            await asyncio.sleep(args.chunk_ms / 1000.0)

    await ws.send(json.dumps({"type": "session.close"}))
    append_jsonl(send_events, {"type": "session.close", "sent_at_s": round(time.monotonic() - start_time, 6)})
    try:
        await asyncio.wait_for(closed.wait(), timeout=args.receive_timeout_s)
    except asyncio.TimeoutError:
        append_jsonl(raw_events, {"receive_timeout_s": args.receive_timeout_s})
    await ws.close()
    receiver_task.cancel()

    generated_wav = out_dir / "generated_target.wav"
    generated_duration_s = pcm16_to_wav(generated_wav, bytes(output_audio), REALTIME_SAMPLE_RATE)
    return {
        "run_id": run["run_id"],
        "backend": "openai_realtime_translation",
        "model": args.model,
        "target_language": target_language,
        "generated_wav_path": str(generated_wav),
        "generated_duration_s": round(generated_duration_s, 6),
        "output_transcript": "".join(output_transcript_parts).strip(),
        "input_transcript": "".join(input_transcript_parts).strip(),
        "raw_events_path": str(raw_events),
        "audio_chunks_path": str(audio_chunks),
        "send_events_path": str(send_events),
        "error_count": len(errors),
    }


def dry_run_output(run: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    stream_duration = float(run["source_stream_duration_s"])
    fake_duration = max(0.5, min(stream_duration, stream_duration * 0.85))
    payload = b"\x00\x00" * int(fake_duration * REALTIME_SAMPLE_RATE)
    generated_wav = out_dir / "generated_target.wav"
    generated_duration_s = pcm16_to_wav(generated_wav, payload, REALTIME_SAMPLE_RATE)
    audio_chunks = out_dir / "audio_chunks.jsonl"
    raw_events = out_dir / "raw_events.jsonl"
    send_events = out_dir / "send_events.jsonl"
    for path in [audio_chunks, raw_events, send_events]:
        if path.exists():
            path.unlink()
    append_jsonl(raw_events, {"type": "dry_run"})
    append_jsonl(
        audio_chunks,
        {
            "chunk_index": 0,
            "arrival_s": 0.0,
            "audio_duration_s": round(generated_duration_s, 6),
            "cumulative_audio_duration_s": round(generated_duration_s, 6),
            "bytes": len(payload),
            "event_type": "dry_run.audio",
        },
    )
    append_jsonl(send_events, {"type": "dry_run"})
    return {
        "run_id": run["run_id"],
        "backend": "dry_run",
        "model": "dry_run",
        "target_language": target_language_for_direction(str(run["direction"])),
        "generated_wav_path": str(generated_wav),
        "generated_duration_s": round(generated_duration_s, 6),
        "output_transcript": str(run.get("target_reference_text") or "")[:500],
        "raw_events_path": str(raw_events),
        "audio_chunks_path": str(audio_chunks),
        "send_events_path": str(send_events),
        "error_count": 0,
    }


async def run_one(run: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir = run_dir(output_dir, run)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if args.resume and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    run = dict(run)
    run.update(prepare_audio(run, out_dir, args.smoke_seconds))
    (out_dir / "run_manifest.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        result = dry_run_output(run, out_dir)
    else:
        result = await run_live(run, out_dir, args)
    result.update(
        {
            "direction": run["direction"],
            "speed_factor": run["speed_factor"],
            "source_eval_wav_path": run["source_eval_wav_path"],
            "source_stream_wav_path": run["source_stream_wav_path"],
            "source_eval_duration_s": run["source_eval_duration_s"],
            "source_stream_duration_s": run["source_stream_duration_s"],
            "target_reference_text": run.get("target_reference_text", ""),
        }
    )
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


async def async_main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = select_runs(read_run_manifest(args.manifest), args)
    results = []
    for index, run in enumerate(runs, start=1):
        result = await run_one(run, output_dir, args)
        results.append(result)
        print(json.dumps({"processed": index, "run_id": result["run_id"]}, ensure_ascii=False), flush=True)
    write_jsonl(output_dir / "results.jsonl", results)
    print(json.dumps({"runs": len(results), "output": str(output_dir)}, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
