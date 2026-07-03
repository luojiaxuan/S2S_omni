#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import websockets
from websockets import Headers

from s2s_omni.floras_live import (
    append_jsonl,
    ffmpeg_convert,
    read_run_manifest,
    sanitize_id,
    source_language_for_direction,
    target_language_for_direction,
)
from s2s_omni.io import write_jsonl
from s2s_omni.rasst import audio_duration_s

SEED_VENDOR = ROOT / "projects" / "acl6060_s2s_metrics_seed" / "vendor" / "seed"
if str(SEED_VENDOR) not in sys.path:
    sys.path.insert(0, str(SEED_VENDOR))

from python_protogen.common.events_pb2 import Type  # noqa: E402
from python_protogen.products.understanding.ast.ast_service_pb2 import (  # noqa: E402
    TranslateRequest,
    TranslateResponse,
)


INPUT_RATE = 16000
CHUNK_SAMPLES = 1600
CHUNK_DURATION_S = 0.1
TARGET_RATE = 24000
BYTES_PER_SAMPLE = 2


@dataclass(frozen=True)
class AstConfig:
    ws_url: str
    api_key: str
    app_key: str
    access_key: str
    resource_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FLORAS live S2S through Seed / ByteDance AST.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--smoke-seconds", type=float, default=0.0)
    parser.add_argument("--source-root", default="")
    parser.add_argument("--ws-url", default="wss://openspeech.bytedance.com/api/v4/ast/v2/translate")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--app-key", default="")
    parser.add_argument("--access-key", default="")
    parser.add_argument("--resource-id", default="volc.service_type.10053")
    parser.add_argument("--receive-timeout-s", type=float, default=60.0)
    parser.add_argument("--progress-interval-s", type=float, default=30.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
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


def resolve_source_path(raw_path: str, source_root: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.exists():
        return path
    if source_root:
        candidate = Path(source_root).expanduser() / path
        if candidate.exists():
            return candidate
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"source wav not found: {raw_path}")


def prepare_audio(run: dict[str, Any], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    source_wav = resolve_source_path(str(run["source_wav_path"]), args.source_root)
    source_eval = out_dir / "source_eval.wav"
    source_stream = out_dir / "source_stream_16k.wav"
    duration = args.smoke_seconds if args.smoke_seconds > 0 else None
    ffmpeg_convert(source_wav, source_eval, sample_rate=INPUT_RATE, speed=1.0, duration_s=duration)
    ffmpeg_convert(
        source_eval,
        source_stream,
        sample_rate=INPUT_RATE,
        speed=float(run["speed_factor"]),
    )
    return {
        "source_wav_path": str(source_wav),
        "source_eval_wav_path": str(source_eval),
        "source_stream_wav_path": str(source_stream),
        "source_eval_duration_s": round(audio_duration_s(source_eval), 6),
        "source_stream_duration_s": round(audio_duration_s(source_stream), 6),
    }


def read_wav_as_chunks(audio_path: str | Path) -> list[bytes]:
    data, sr = sf.read(str(audio_path), dtype="int16", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True).astype(np.int16)
    data = data[:, 0]
    if sr != INPUT_RATE:
        n = len(data)
        new_n = int(n * INPUT_RATE / sr)
        data = np.interp(
            np.linspace(0, n - 1, new_n),
            np.arange(n),
            data.astype(np.float64),
        ).astype(np.int16)
    return [data[i : i + CHUNK_SAMPLES].tobytes() for i in range(0, len(data), CHUNK_SAMPLES) if data[i : i + CHUNK_SAMPLES].size]


def save_pcm_as_wav(
    pcm_bytes: bytes,
    wav_path: Path,
    *,
    sample_rate: int = TARGET_RATE,
    channels: int = 1,
) -> float:
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(BYTES_PER_SAMPLE)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return len(pcm_bytes) / (sample_rate * BYTES_PER_SAMPLE * channels)


def _base_req(session_id: str, event: int, src_lang: str, tgt_lang: str) -> TranslateRequest:
    req = TranslateRequest()
    req.request_meta.SessionID = session_id
    req.event = event
    req.user.uid = "s2s_omni_seed_floras"
    req.user.did = "s2s_omni_seed_floras"
    req.source_audio.format = "wav"
    req.source_audio.rate = INPUT_RATE
    req.source_audio.bits = 16
    req.source_audio.channel = 1
    req.target_audio.format = "pcm_s16le"
    req.target_audio.rate = TARGET_RATE
    req.request.mode = "s2s"
    req.request.source_language = src_lang
    req.request.target_language = tgt_lang
    return req


def make_start(session_id: str, src_lang: str, tgt_lang: str) -> bytes:
    return _base_req(session_id, Type.StartSession, src_lang, tgt_lang).SerializeToString()


def make_chunk(session_id: str, chunk: bytes, src_lang: str, tgt_lang: str) -> bytes:
    req = _base_req(session_id, Type.TaskRequest, src_lang, tgt_lang)
    req.source_audio.binary_data = chunk
    return req.SerializeToString()


def make_finish(session_id: str, src_lang: str, tgt_lang: str) -> bytes:
    return _base_req(session_id, Type.FinishSession, src_lang, tgt_lang).SerializeToString()


def build_headers(conf: AstConfig, conn_id: str) -> Headers:
    if conf.api_key:
        return Headers(
            {
                "X-Api-Key": conf.api_key,
                "X-Api-Resource-Id": conf.resource_id,
                "X-Api-Connect-Id": conn_id,
            }
        )
    return Headers(
        {
            "X-Api-App-Key": conf.app_key,
            "X-Api-Access-Key": conf.access_key,
            "X-Api-Resource-Id": conf.resource_id,
            "X-Api-Connect-Id": conn_id,
        }
    )


async def connect_ws(conf: AstConfig, conn_id: str):
    try:
        return await websockets.connect(
            conf.ws_url,
            additional_headers=build_headers(conf, conn_id),
            max_size=1_000_000_000,
            ping_interval=None,
        )
    except TypeError:
        return await websockets.connect(
            conf.ws_url,
            extra_headers=build_headers(conf, conn_id),
            max_size=1_000_000_000,
            ping_interval=None,
        )


def sanitize_event(resp: TranslateResponse, received_at_s: float) -> dict[str, Any]:
    return {
        "event": int(resp.event),
        "event_name": Type.Name(resp.event) if resp.event in Type.values() else str(resp.event),
        "received_at_s": round(received_at_s, 6),
        "data_bytes": len(resp.data),
        "text_chars": len(resp.text or ""),
        "start_time": int(resp.start_time),
        "end_time": int(resp.end_time),
        "message": str(resp.response_meta.Message or ""),
    }


async def run_live(run: dict[str, Any], out_dir: Path, args: argparse.Namespace, conf: AstConfig) -> dict[str, Any]:
    raw_events = out_dir / "raw_events.jsonl"
    audio_chunks = out_dir / "audio_chunks.jsonl"
    send_events = out_dir / "send_events.jsonl"
    for path in [raw_events, audio_chunks, send_events]:
        if path.exists():
            path.unlink()

    src_lang = source_language_for_direction(str(run["direction"]))
    tgt_lang = target_language_for_direction(str(run["direction"]))
    audio_input = read_wav_as_chunks(run["source_stream_wav_path"])
    conn_id = str(uuid.uuid4())
    conn = await connect_ws(conf, conn_id)
    session_id = str(uuid.uuid4())
    await conn.send(make_start(session_id, src_lang, tgt_lang))
    init_raw = await conn.recv()
    init_resp = TranslateResponse()
    init_resp.ParseFromString(init_raw)
    if init_resp.event != Type.SessionStarted:
        await conn.close()
        raise RuntimeError(f"Seed AST session failed: event={init_resp.event} msg={init_resp.response_meta.Message}")

    start_time = time.monotonic()
    first_send_time = start_time
    output_audio = bytearray()
    transcript_parts: list[str] = []
    errors: list[dict[str, Any]] = []
    cumulative_audio_s = 0.0
    output_chunk_index = 0
    next_progress_s = args.progress_interval_s

    async def send_audio() -> None:
        for index, chunk in enumerate(audio_input):
            target_time = first_send_time + index * CHUNK_DURATION_S
            now = time.monotonic()
            if now < target_time:
                await asyncio.sleep(target_time - now)
            await conn.send(make_chunk(session_id, chunk, src_lang, tgt_lang))
            append_jsonl(
                send_events,
                {
                    "send_index": index,
                    "sent_at_s": round(time.monotonic() - start_time, 6),
                    "audio_duration_s": round(len(chunk) / (INPUT_RATE * BYTES_PER_SAMPLE), 6),
                    "bytes": len(chunk),
                },
            )
        await conn.send(make_finish(session_id, src_lang, tgt_lang))
        append_jsonl(send_events, {"type": "FinishSession", "sent_at_s": round(time.monotonic() - start_time, 6)})

    sender = asyncio.create_task(send_audio())
    receive_deadline: float | None = None
    try:
        while True:
            if receive_deadline is None:
                raw = await conn.recv()
            else:
                remaining_s = receive_deadline - time.monotonic()
                if remaining_s <= 0:
                    append_jsonl(raw_events, {"receive_timeout_s": args.receive_timeout_s})
                    break
                raw = await asyncio.wait_for(conn.recv(), timeout=remaining_s)
            now_s = time.monotonic() - start_time
            resp = TranslateResponse()
            resp.ParseFromString(raw)
            append_jsonl(raw_events, sanitize_event(resp, now_s))
            if resp.event in (Type.SessionFailed, Type.SessionCanceled):
                errors.append({"event": int(resp.event), "message": str(resp.response_meta.Message or "")})
                break
            if resp.event == Type.SessionFinished:
                break
            if resp.data:
                duration_s = len(resp.data) / (TARGET_RATE * BYTES_PER_SAMPLE)
                cumulative_audio_s += duration_s
                output_audio.extend(resp.data)
                append_jsonl(
                    audio_chunks,
                    {
                        "chunk_index": output_chunk_index,
                        "arrival_s": round(now_s, 6),
                        "audio_duration_s": round(duration_s, 6),
                        "cumulative_audio_duration_s": round(cumulative_audio_s, 6),
                        "bytes": len(resp.data),
                        "event_type": Type.Name(resp.event) if resp.event in Type.values() else str(resp.event),
                    },
                )
                output_chunk_index += 1
            if resp.text and resp.event == Type.TranslationSubtitleEnd:
                transcript_parts.append(resp.text)
            if sender.done() and receive_deadline is None:
                receive_deadline = time.monotonic() + args.receive_timeout_s
            if args.progress_interval_s > 0 and now_s >= next_progress_s:
                print(
                    json.dumps(
                        {
                            "run_id": run["run_id"],
                            "phase": "receiving",
                            "elapsed_s": round(now_s, 3),
                            "output_audio_s": round(cumulative_audio_s, 3),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                next_progress_s += args.progress_interval_s
    finally:
        if not sender.done():
            sender.cancel()
        try:
            await sender
        except asyncio.CancelledError:
            pass
        await conn.close()

    generated_wav = out_dir / "generated_target.wav"
    generated_duration_s = save_pcm_as_wav(bytes(output_audio), generated_wav)
    subtitle = "\n".join(text for text in transcript_parts if text)
    (out_dir / "ast_translation_subtitle.txt").write_text(subtitle, encoding="utf-8")
    return {
        "run_id": run["run_id"],
        "backend": "seed_ast_s2s",
        "model": "seed_ast",
        "target_language": tgt_lang,
        "generated_wav_path": str(generated_wav),
        "generated_duration_s": round(generated_duration_s, 6),
        "output_transcript": "",
        "ast_translation_subtitle": subtitle,
        "ast_translation_subtitle_path": str(out_dir / "ast_translation_subtitle.txt"),
        "raw_events_path": str(raw_events),
        "audio_chunks_path": str(audio_chunks),
        "send_events_path": str(send_events),
        "error_count": len(errors),
        "errors": errors,
    }


def dry_run_output(run: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    duration_s = max(0.5, min(float(run["source_stream_duration_s"]), 2.0))
    payload = b"\x00\x00" * int(duration_s * TARGET_RATE)
    generated_wav = out_dir / "generated_target.wav"
    generated_duration_s = save_pcm_as_wav(payload, generated_wav)
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
        "backend": "seed_ast_dry_run",
        "model": "dry_run",
        "target_language": target_language_for_direction(str(run["direction"])),
        "generated_wav_path": str(generated_wav),
        "generated_duration_s": round(generated_duration_s, 6),
        "output_transcript": "",
        "raw_events_path": str(raw_events),
        "audio_chunks_path": str(audio_chunks),
        "send_events_path": str(send_events),
        "error_count": 0,
    }


async def run_one(run: dict[str, Any], output_dir: Path, args: argparse.Namespace, conf: AstConfig) -> dict[str, Any]:
    out_dir = run_dir(output_dir, run)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if args.resume and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    run = dict(run)
    run.update(prepare_audio(run, out_dir, args))
    (out_dir / "run_manifest.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    result = dry_run_output(run, out_dir) if args.dry_run else await run_live(run, out_dir, args, conf)
    result.update(
        {
            "direction": run["direction"],
            "speed_factor": run["speed_factor"],
            "source_eval_wav_path": run["source_eval_wav_path"],
            "source_stream_wav_path": run["source_stream_wav_path"],
            "source_eval_duration_s": run["source_eval_duration_s"],
            "source_stream_duration_s": run["source_stream_duration_s"],
            "target_reference_text": run.get("target_reference_text", ""),
            "chunk_ms": int(CHUNK_DURATION_S * 1000),
        }
    )
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


async def async_main() -> None:
    args = parse_args()
    if not args.dry_run and not args.api_key and not (args.app_key and args.access_key):
        raise SystemExit("pass --api-key, or pass both --app-key and --access-key")
    conf = AstConfig(
        ws_url=args.ws_url,
        api_key=args.api_key,
        app_key=args.app_key,
        access_key=args.access_key,
        resource_id=args.resource_id,
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = select_runs(read_run_manifest(args.manifest), args)
    results = []
    for index, run in enumerate(runs, start=1):
        result = await run_one(run, output_dir, args, conf)
        results.append(result)
        print(json.dumps({"processed": index, "run_id": result["run_id"]}, ensure_ascii=False), flush=True)
    write_jsonl(output_dir / "results.jsonl", results)
    print(json.dumps({"runs": len(results), "output": str(output_dir)}, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
