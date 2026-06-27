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
    GEMINI_INPUT_SAMPLE_RATE,
    GEMINI_TRANSLATION_MODEL,
    GEMINI_TRANSLATION_WS,
    REALTIME_CHUNK_MS,
    REALTIME_SAMPLE_RATE,
    append_jsonl,
    env_api_key,
    ffmpeg_convert,
    pcm16_duration_s,
    pcm16_to_wav,
    read_run_manifest,
    sanitize_id,
    wav_to_pcm16,
)
from s2s_omni.io import write_jsonl
from s2s_omni.rasst import audio_duration_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FLORAS live S2S through Gemini Live Translate."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--smoke-seconds", type=float, default=0.0)
    parser.add_argument("--chunk-ms", type=int, default=REALTIME_CHUNK_MS)
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_LIVE_TRANSLATE_MODEL", GEMINI_TRANSLATION_MODEL),
    )
    parser.add_argument(
        "--ws-url",
        default=os.environ.get("GEMINI_LIVE_TRANSLATE_WS", GEMINI_TRANSLATION_WS),
    )
    parser.add_argument("--receive-timeout-s", type=float, default=60.0)
    parser.add_argument("--post-send-idle-s", type=float, default=8.0)
    parser.add_argument("--setup-timeout-s", type=float, default=30.0)
    parser.add_argument("--progress-interval-s", type=float, default=30.0)
    parser.add_argument("--pace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--echo-target-language",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--target-language-code", default="")
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


def gemini_target_language_code(direction: str) -> str:
    if direction == "zh-en":
        return "en"
    if direction == "en-zh":
        return "zh-Hans"
    raise ValueError(f"unsupported direction: {direction}")


def prepare_audio(run: dict[str, Any], out_dir: Path, smoke_seconds: float) -> dict[str, Any]:
    speed = float(run["speed_factor"])
    source_wav = Path(str(run["source_wav_path"]))
    source_eval = out_dir / "source_eval.wav"
    source_stream = out_dir / "source_stream_16k.wav"
    duration = smoke_seconds if smoke_seconds > 0 else None
    ffmpeg_convert(
        source_wav,
        source_eval,
        sample_rate=GEMINI_INPUT_SAMPLE_RATE,
        speed=1.0,
        duration_s=duration,
    )
    ffmpeg_convert(
        source_eval,
        source_stream,
        sample_rate=GEMINI_INPUT_SAMPLE_RATE,
        speed=speed,
    )
    return {
        "source_eval_wav_path": str(source_eval),
        "source_stream_wav_path": str(source_stream),
        "source_eval_duration_s": round(audio_duration_s(source_eval), 6),
        "source_stream_duration_s": round(audio_duration_s(source_stream), 6),
    }


def model_name(model: str) -> str:
    return model if model.startswith("models/") else f"models/{model}"


def setup_payload(args: argparse.Namespace, target_language_code: str) -> dict[str, Any]:
    return {
        "setup": {
            "model": model_name(args.model),
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
                "translationConfig": {
                    "targetLanguageCode": target_language_code,
                    "echoTargetLanguage": bool(args.echo_target_language),
                },
            },
        }
    }


def audio_input_payload(payload: bytes) -> dict[str, Any]:
    return {
        "realtimeInput": {
            "audio": {
                "data": base64.b64encode(payload).decode("ascii"),
                "mimeType": f"audio/pcm;rate={GEMINI_INPUT_SAMPLE_RATE}",
            }
        }
    }


def sanitize_event(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_event_item(key, item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_event(item) for item in value]
    return value


def sanitize_event_item(key: str, value: Any) -> Any:
    if key == "data" and isinstance(value, str) and len(value) > 256:
        return f"<base64:{len(value)} chars>"
    return sanitize_event(value)


def inline_audio_payloads(event: dict[str, Any]) -> list[bytes]:
    out: list[bytes] = []
    content = event.get("serverContent")
    if not isinstance(content, dict):
        return out
    turn = content.get("modelTurn")
    if not isinstance(turn, dict):
        return out
    parts = turn.get("parts")
    if not isinstance(parts, list):
        return out
    for part in parts:
        if not isinstance(part, dict):
            continue
        inline = part.get("inlineData")
        if not isinstance(inline, dict):
            continue
        mime_type = str(inline.get("mimeType") or "")
        data = inline.get("data")
        if not isinstance(data, str) or not data:
            continue
        if mime_type and not mime_type.startswith("audio/"):
            continue
        try:
            out.append(base64.b64decode(data))
        except Exception:
            continue
    return out


def transcription_text(event: dict[str, Any], key: str) -> str:
    content = event.get("serverContent")
    if not isinstance(content, dict):
        return ""
    transcription = content.get(key)
    if not isinstance(transcription, dict):
        return ""
    text = transcription.get("text")
    return text if isinstance(text, str) else ""


async def connect_ws(url: str):
    import websockets

    try:
        return await websockets.connect(url, max_size=None)
    except TypeError:
        return await websockets.connect(url)


async def run_live(run: dict[str, Any], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    raw_events = out_dir / "raw_events.jsonl"
    audio_chunks = out_dir / "audio_chunks.jsonl"
    send_events = out_dir / "send_events.jsonl"
    for path in [raw_events, audio_chunks, send_events]:
        if path.exists():
            path.unlink()

    target_language_code = args.target_language_code or gemini_target_language_code(
        str(run["direction"])
    )
    api_key = env_api_key("GEMINI_API_KEY")
    url = f"{args.ws_url}?key={api_key}"

    ws = await connect_ws(url)
    output_audio = bytearray()
    output_transcript_parts: list[str] = []
    input_transcript_parts: list[str] = []
    errors: list[dict[str, Any]] = []
    setup_done = asyncio.Event()
    receiver_done = asyncio.Event()
    start_time = time.monotonic()
    last_event_time = start_time
    cumulative_audio_s = 0.0
    chunk_index = 0

    async def receiver() -> None:
        nonlocal chunk_index, cumulative_audio_s, last_event_time
        try:
            async for message in ws:
                now = time.monotonic()
                last_event_time = now
                now_s = now - start_time
                if isinstance(message, bytes):
                    try:
                        message = message.decode("utf-8")
                    except UnicodeDecodeError:
                        append_jsonl(
                            raw_events,
                            {
                                "received_at_s": round(now_s, 6),
                                "binary_message_bytes": len(message),
                            },
                        )
                        continue
                event = json.loads(message)
                safe = sanitize_event(event)
                if isinstance(safe, dict):
                    safe["received_at_s"] = round(now_s, 6)
                    append_jsonl(raw_events, safe)
                if event.get("setupComplete") is not None:
                    setup_done.set()
                if event.get("error"):
                    errors.append(safe if isinstance(safe, dict) else {"error": safe})
                for payload in inline_audio_payloads(event):
                    duration_s = pcm16_duration_s(payload, REALTIME_SAMPLE_RATE)
                    cumulative_audio_s += duration_s
                    output_audio.extend(payload)
                    append_jsonl(
                        audio_chunks,
                        {
                            "chunk_index": chunk_index,
                            "arrival_s": round(now_s, 6),
                            "audio_duration_s": round(duration_s, 6),
                            "cumulative_audio_duration_s": round(cumulative_audio_s, 6),
                            "bytes": len(payload),
                            "event_type": "serverContent.modelTurn.inlineData",
                        },
                    )
                    chunk_index += 1
                output_delta = transcription_text(event, "outputTranscription")
                if output_delta:
                    output_transcript_parts.append(output_delta)
                input_delta = transcription_text(event, "inputTranscription")
                if input_delta:
                    input_transcript_parts.append(input_delta)
        except Exception as exc:
            append_jsonl(raw_events, {"receiver_error": f"{type(exc).__name__}: {exc}"})
        finally:
            receiver_done.set()

    receiver_task = asyncio.create_task(receiver())
    await ws.send(json.dumps(setup_payload(args, target_language_code)))
    append_jsonl(
        send_events,
        {
            "type": "setup",
            "sent_at_s": 0.0,
            "model": args.model,
            "target_language_code": target_language_code,
        },
    )

    try:
        await asyncio.wait_for(setup_done.wait(), timeout=args.setup_timeout_s)
    except asyncio.TimeoutError:
        append_jsonl(raw_events, {"setup_timeout_s": args.setup_timeout_s})

    pcm = wav_to_pcm16(run["source_stream_wav_path"], GEMINI_INPUT_SAMPLE_RATE)
    samples_per_chunk = int(round(GEMINI_INPUT_SAMPLE_RATE * args.chunk_ms / 1000.0))
    bytes_per_chunk = samples_per_chunk * 2
    total_stream_s = pcm16_duration_s(pcm, GEMINI_INPUT_SAMPLE_RATE)
    send_index = 0
    next_progress_s = args.progress_interval_s
    for start in range(0, len(pcm), bytes_per_chunk):
        payload = pcm[start : start + bytes_per_chunk]
        await ws.send(json.dumps(audio_input_payload(payload)))
        sent_at_s = time.monotonic() - start_time
        append_jsonl(
            send_events,
            {
                "type": "realtime_input.audio",
                "send_index": send_index,
                "sent_at_s": round(sent_at_s, 6),
                "audio_duration_s": round(pcm16_duration_s(payload, GEMINI_INPUT_SAMPLE_RATE), 6),
                "bytes": len(payload),
            },
        )
        send_index += 1
        elapsed_s = time.monotonic() - start_time
        if args.progress_interval_s > 0 and elapsed_s >= next_progress_s:
            print(
                json.dumps(
                    {
                        "run_id": run["run_id"],
                        "phase": "sending",
                        "elapsed_s": round(elapsed_s, 3),
                        "sent_stream_s": round(
                            min(
                                total_stream_s,
                                pcm16_duration_s(
                                    pcm[: start + len(payload)],
                                    GEMINI_INPUT_SAMPLE_RATE,
                                ),
                            ),
                            3,
                        ),
                        "total_stream_s": round(total_stream_s, 3),
                        "output_audio_s": round(cumulative_audio_s, 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            next_progress_s += args.progress_interval_s
        if args.pace:
            await asyncio.sleep(args.chunk_ms / 1000.0)

    send_finished_at = time.monotonic()
    append_jsonl(
        send_events,
        {
            "type": "input_finished",
            "sent_at_s": round(send_finished_at - start_time, 6),
        },
    )
    deadline = send_finished_at + args.receive_timeout_s
    while time.monotonic() < deadline and not receiver_done.is_set():
        idle_base = max(last_event_time, send_finished_at)
        idle_s = time.monotonic() - idle_base
        if idle_s >= args.post_send_idle_s:
            break
        await asyncio.sleep(min(0.5, max(0.05, args.post_send_idle_s - idle_s)))

    try:
        await ws.close()
    finally:
        receiver_task.cancel()

    generated_wav = out_dir / "generated_target.wav"
    generated_duration_s = pcm16_to_wav(generated_wav, bytes(output_audio), REALTIME_SAMPLE_RATE)
    return {
        "run_id": run["run_id"],
        "backend": "gemini_live_translate",
        "model": args.model,
        "target_language": target_language_code,
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
        "target_language": gemini_target_language_code(str(run["direction"])),
        "generated_wav_path": str(generated_wav),
        "generated_duration_s": round(generated_duration_s, 6),
        "output_transcript": str(run.get("target_reference_text") or "")[:500],
        "raw_events_path": str(raw_events),
        "audio_chunks_path": str(audio_chunks),
        "send_events_path": str(send_events),
        "error_count": 0,
    }


async def run_one(
    run: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir = run_dir(output_dir, run)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if args.resume and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    run = dict(run)
    run.update(prepare_audio(run, out_dir, args.smoke_seconds))
    (out_dir / "run_manifest.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
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
        print(
            json.dumps(
                {"processed": index, "run_id": result["run_id"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
    write_jsonl(output_dir / "results.jsonl", results)
    print(
        json.dumps(
            {"runs": len(results), "output": str(output_dir)},
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
