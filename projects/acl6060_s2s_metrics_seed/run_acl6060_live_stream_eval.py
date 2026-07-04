#!/usr/bin/env python3
"""Run ACL6060 full-wav streaming eval through OpenAI Realtime or Gemini Live."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import time
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HF_REPO = "gavinlaw/rasst-main-result-data"
OPENAI_MODEL = "gpt-realtime-translate"
OPENAI_WS_URL = "wss://api.openai.com/v1/realtime/translations"
GEMINI_MODEL = "gemini-3.5-live-translate-preview"
GEMINI_WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
OPENAI_SAMPLE_RATE = 24000
GEMINI_SAMPLE_RATE = 16000
TARGET_LANGUAGES = {"zh": "Chinese", "ja": "Japanese", "de": "German"}
HF_ACL6060_ZH_FILES = [
    "main_result/inputs/acl_zh/audio.yaml",
    "main_result/inputs/acl_zh/ref.txt",
    "main_result/inputs/acl_zh/source.list",
    "main_result/inputs/acl_zh/source_text.txt",
    "main_result/inputs/acl_zh/target.list",
    "main_result/audio/acl6060/2022.acl-long.110.wav",
    "main_result/audio/acl6060/2022.acl-long.117.wav",
    "main_result/audio/acl6060/2022.acl-long.268.wav",
    "main_result/audio/acl6060/2022.acl-long.367.wav",
    "main_result/audio/acl6060/2022.acl-long.590.wav",
]


@dataclass(frozen=True)
class ReleasePaths:
    input_dir: Path
    audio_root: Path
    source_list: Path
    target_list: Path
    ref_file: Path
    source_text_file: Path
    audio_yaml: Path


@dataclass
class StreamedText:
    text_parts: list[str]
    delays_ms: list[float]
    elapsed_ms: list[float]
    input_transcript_parts: list[str]
    errors: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream the ACL6060 main-result full wavs into a live API and write "
            "RASST/SimulEval-style instances.log records."
        )
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--provider", required=True, choices=["openai", "gemini"])
    parser.add_argument("--api-key-file", required=True, type=Path)
    parser.add_argument("--model", default="")
    parser.add_argument("--target-lang", default="zh", choices=sorted(TARGET_LANGUAGES))
    parser.add_argument("--chunk-ms", type=int, default=960)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-audio-seconds", type=float, default=0.0)
    parser.add_argument("--receive-timeout-s", type=float, default=60.0)
    parser.add_argument("--post-send-idle-s", type=float, default=8.0)
    parser.add_argument("--setup-timeout-s", type=float, default=30.0)
    parser.add_argument("--max-session-input-s", type=float, default=0.0)
    parser.add_argument("--progress-interval-s", type=float, default=30.0)
    parser.add_argument("--pace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download-hf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--openai-ws-url", default=OPENAI_WS_URL)
    parser.add_argument("--gemini-ws-url", default=GEMINI_WS_URL)
    parser.add_argument("--instructions", default="")
    return parser.parse_args()


def read_secret(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"empty API key file: {path}")
    return value


def hf_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{path}?download=true"


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(output_path)


def download_hf_subset(dataset_root: Path) -> None:
    for rel_path in HF_ACL6060_ZH_FILES:
        out_path = dataset_root / rel_path
        print(
            json.dumps({"download": rel_path, "path": str(out_path)}, ensure_ascii=False),
            flush=True,
        )
        download_file(hf_url(rel_path), out_path)


def resolve_paths(dataset_root: Path, target_lang: str) -> ReleasePaths:
    input_name = f"acl_{target_lang}"
    candidates = [
        dataset_root / "main_result",
        dataset_root / "data" / "main_result",
    ]
    for base in candidates:
        input_dir = base / "inputs" / input_name
        audio_root = base / "audio" / "acl6060"
        if input_dir.is_dir() and audio_root.is_dir():
            return ReleasePaths(
                input_dir=input_dir,
                audio_root=audio_root,
                source_list=input_dir / "source.list",
                target_list=input_dir / "target.list",
                ref_file=input_dir / "ref.txt",
                source_text_file=input_dir / "source_text.txt",
                audio_yaml=input_dir / "audio.yaml",
            )
    raise FileNotFoundError(
        f"could not find main_result inputs/audio under {dataset_root}; "
        "pass --download-hf or point --dataset-root at a RASST release data root"
    )


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def parse_simple_audio_yaml(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            if current is not None:
                records.append(current)
            current = {}
            stripped = stripped[2:].strip()
            if not stripped:
                continue
        if current is None or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        current[key.strip()] = parse_scalar(value.strip())
    if current is not None:
        records.append(current)
    return records


def parse_scalar(value: str) -> Any:
    value = value.strip().strip("'\"")
    if value == "":
        return ""
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def resolve_audio_path(raw_path: str, dataset_root: Path, paths: ReleasePaths) -> Path:
    source = Path(raw_path.strip())
    if source.is_absolute() and source.exists():
        return source
    candidates = [
        dataset_root / source,
        dataset_root / "data" / source,
        dataset_root / source.as_posix().removeprefix("data/"),
        paths.audio_root / source.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"source wav not found for {raw_path!r}")


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")


def ffmpeg_convert(
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    duration_s: float,
) -> None:
    ensure_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
    ]
    if duration_s > 0:
        cmd.extend(["-t", f"{duration_s:.6f}"])
    cmd.extend(["-ac", "1", "-ar", str(sample_rate), "-sample_fmt", "s16", str(output_path)])
    subprocess.run(cmd, check=True)


def wav_duration_ms(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return round(handle.getnframes() * 1000.0 / float(handle.getframerate()), 3)


def wav_to_pcm16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        if channels != 1 or width != 2:
            raise ValueError(f"{path} must be mono signed 16-bit PCM")
        return handle.readframes(handle.getnframes())


def pcm16_duration_s(payload: bytes, sample_rate: int) -> float:
    return len(payload) / 2.0 / float(sample_rate)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def load_done_indices(instances_path: Path) -> set[int]:
    if not instances_path.exists():
        return set()
    done: set[int] = set()
    for line in instances_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(int(json.loads(line)["index"]))
    return done


def count_latency_units(text: str, target_lang: str) -> int:
    if target_lang in {"zh", "ja"}:
        return sum(1 for ch in text if not ch.isspace())
    return len(text.split())


def text_units(text: str, target_lang: str) -> list[str]:
    if target_lang in {"zh", "ja"}:
        return [ch for ch in text if not ch.isspace()]
    return text.split()


def add_text_delta(
    streamed: StreamedText,
    delta: str,
    target_lang: str,
    delay_ms: float,
    elapsed_ms: float,
) -> None:
    if not delta:
        return
    streamed.text_parts.append(delta)
    unit_count = len(text_units(delta, target_lang))
    if unit_count <= 0:
        return
    streamed.delays_ms.extend([round(delay_ms, 3)] * unit_count)
    streamed.elapsed_ms.extend([round(elapsed_ms, 3)] * unit_count)


def sanitize_event(event: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    event_type = str(event.get("type") or "")
    for key, value in event.items():
        if isinstance(value, str) and key in {"audio", "delta", "data"} and "audio" in event_type:
            out[key] = f"<base64:{len(value)} chars>"
        else:
            out[key] = value
    return out


def openai_output_delta(event: dict[str, Any]) -> str:
    if str(event.get("type") or "") != "session.output_transcript.delta":
        return ""
    for key in ["delta", "text", "transcript"]:
        value = event.get(key)
        if isinstance(value, str):
            return value
    return ""


def openai_input_delta(event: dict[str, Any]) -> str:
    if str(event.get("type") or "") != "session.input_transcript.delta":
        return ""
    value = event.get("delta")
    return value if isinstance(value, str) else ""


async def connect_openai(url: str, api_key: str, model: str):
    import websockets

    headers = [
        ("Authorization", f"Bearer {api_key}"),
        ("OpenAI-Safety-Identifier", "s2s-omni-acl6060-streaming"),
    ]
    full_url = f"{url}?model={model}"
    try:
        return await websockets.connect(
            full_url,
            additional_headers=headers,
            max_size=None,
            ping_interval=None,
        )
    except TypeError:
        return await websockets.connect(
            full_url,
            extra_headers=headers,
            max_size=None,
            ping_interval=None,
        )


def openai_session_update(args: argparse.Namespace) -> dict[str, Any]:
    session: dict[str, Any] = {"audio": {"output": {"language": args.target_lang}}}
    if args.instructions:
        session["instructions"] = args.instructions
    return {"type": "session.update", "session": session}


async def run_openai_stream(
    pcm: bytes,
    sample_rate: int,
    api_key: str,
    out_dir: Path,
    args: argparse.Namespace,
    total_source_ms: float,
) -> StreamedText:
    ws = await connect_openai(args.openai_ws_url, api_key, args.model or OPENAI_MODEL)
    raw_events = out_dir / "raw_events.jsonl"
    send_events = out_dir / "send_events.jsonl"
    for path in [raw_events, send_events]:
        if path.exists():
            path.unlink()

    streamed = StreamedText([], [], [], [], [])
    start_time = time.monotonic()
    closed = asyncio.Event()
    sent_audio_s = 0.0

    async def receiver() -> None:
        nonlocal sent_audio_s
        try:
            while not closed.is_set():
                message = await ws.recv()
                now_s = time.monotonic() - start_time
                event = json.loads(message)
                safe = sanitize_event(event)
                safe["received_at_s"] = round(now_s, 6)
                safe["sent_source_ms"] = round(min(total_source_ms, sent_audio_s * 1000.0), 3)
                append_jsonl(raw_events, safe)
                event_type = str(event.get("type") or "")
                if "error" in event_type or event.get("error"):
                    streamed.errors.append(safe)
                add_text_delta(
                    streamed,
                    openai_output_delta(event),
                    args.target_lang,
                    delay_ms=min(total_source_ms, sent_audio_s * 1000.0),
                    elapsed_ms=now_s * 1000.0,
                )
                input_delta = openai_input_delta(event)
                if input_delta:
                    streamed.input_transcript_parts.append(input_delta)
                if event_type in {"session.closed", "translation_session.closed"}:
                    closed.set()
        except Exception as exc:
            append_jsonl(raw_events, {"receiver_error": f"{type(exc).__name__}: {exc}"})
            closed.set()

    receiver_task = asyncio.create_task(receiver())
    await ws.send(json.dumps(openai_session_update(args)))
    append_jsonl(send_events, {"type": "session.update", "sent_at_s": 0.0})

    bytes_per_chunk = max(2, int(round(sample_rate * args.chunk_ms / 1000.0)) * 2)
    next_progress_s = args.progress_interval_s
    for send_index, start in enumerate(range(0, len(pcm), bytes_per_chunk)):
        payload = pcm[start : start + bytes_per_chunk]
        await ws.send(
            json.dumps(
                {
                    "type": "session.input_audio_buffer.append",
                    "audio": base64.b64encode(payload).decode("ascii"),
                }
            )
        )
        sent_audio_s = pcm16_duration_s(pcm[: start + len(payload)], sample_rate)
        elapsed_s = time.monotonic() - start_time
        append_jsonl(
            send_events,
            {
                "type": "session.input_audio_buffer.append",
                "send_index": send_index,
                "sent_at_s": round(elapsed_s, 6),
                "sent_source_ms": round(min(total_source_ms, sent_audio_s * 1000.0), 3),
                "audio_duration_s": round(pcm16_duration_s(payload, sample_rate), 6),
                "bytes": len(payload),
            },
        )
        if args.progress_interval_s > 0 and elapsed_s >= next_progress_s:
            print_progress(
                args,
                out_dir,
                "sending",
                elapsed_s,
                sent_audio_s,
                len(streamed.delays_ms),
            )
            next_progress_s += args.progress_interval_s
        if args.pace:
            await asyncio.sleep(args.chunk_ms / 1000.0)

    await ws.send(json.dumps({"type": "session.close"}))
    append_jsonl(
        send_events,
        {"type": "session.close", "sent_at_s": round(time.monotonic() - start_time, 6)},
    )
    deadline = time.monotonic() + args.receive_timeout_s
    while not closed.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    await ws.close()
    receiver_task.cancel()
    return streamed


def gemini_model_name(model: str) -> str:
    return model if model.startswith("models/") else f"models/{model}"


def gemini_setup_payload(args: argparse.Namespace) -> dict[str, Any]:
    target_language_code = "zh-Hans" if args.target_lang == "zh" else args.target_lang
    return {
        "setup": {
            "model": gemini_model_name(args.model or GEMINI_MODEL),
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "translationConfig": {
                    "targetLanguageCode": target_language_code,
                    "echoTargetLanguage": False,
                },
            },
        }
    }


def gemini_audio_payload(payload: bytes) -> dict[str, Any]:
    return {
        "realtimeInput": {
            "audio": {
                "data": base64.b64encode(payload).decode("ascii"),
                "mimeType": f"audio/pcm;rate={GEMINI_SAMPLE_RATE}",
            }
        }
    }


def gemini_transcription(event: dict[str, Any], key: str) -> str:
    content = event.get("serverContent")
    if not isinstance(content, dict):
        return ""
    transcription = content.get(key)
    if not isinstance(transcription, dict):
        return ""
    text = transcription.get("text")
    return text if isinstance(text, str) else ""


def sanitize_gemini(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_gemini_item(key, item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_gemini(item) for item in value]
    return value


def sanitize_gemini_item(key: str, value: Any) -> Any:
    if key == "data" and isinstance(value, str) and len(value) > 256:
        return f"<base64:{len(value)} chars>"
    return sanitize_gemini(value)


async def connect_gemini(url: str, api_key: str):
    import websockets

    return await websockets.connect(f"{url}?key={api_key}", max_size=None, ping_interval=None)


async def run_gemini_stream(
    pcm: bytes,
    sample_rate: int,
    api_key: str,
    out_dir: Path,
    args: argparse.Namespace,
    total_source_ms: float,
    *,
    base_source_ms: float = 0.0,
    segment_index: int = 0,
    run_start_time: float | None = None,
    reset_logs: bool = True,
) -> StreamedText:
    ws = await connect_gemini(args.gemini_ws_url, api_key)
    raw_events = out_dir / "raw_events.jsonl"
    send_events = out_dir / "send_events.jsonl"
    if reset_logs:
        for path in [raw_events, send_events]:
            if path.exists():
                path.unlink()

    streamed = StreamedText([], [], [], [], [])
    setup_done = asyncio.Event()
    receiver_done = asyncio.Event()
    start_time = run_start_time if run_start_time is not None else time.monotonic()
    last_event_time = time.monotonic()
    sent_audio_s = 0.0

    async def receiver() -> None:
        nonlocal last_event_time, sent_audio_s
        try:
            async for message in ws:
                now = time.monotonic()
                last_event_time = now
                now_s = now - start_time
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                event = json.loads(message)
                safe = sanitize_gemini(event)
                if isinstance(safe, dict):
                    safe["received_at_s"] = round(now_s, 6)
                    safe["segment_index"] = segment_index
                    safe["sent_source_ms"] = round(
                        min(total_source_ms, base_source_ms + sent_audio_s * 1000.0),
                        3,
                    )
                    append_jsonl(raw_events, safe)
                if event.get("setupComplete") is not None:
                    setup_done.set()
                if event.get("error"):
                    streamed.errors.append(safe if isinstance(safe, dict) else {"error": safe})
                add_text_delta(
                    streamed,
                    gemini_transcription(event, "outputTranscription"),
                    args.target_lang,
                    delay_ms=min(total_source_ms, base_source_ms + sent_audio_s * 1000.0),
                    elapsed_ms=now_s * 1000.0,
                )
                input_delta = gemini_transcription(event, "inputTranscription")
                if input_delta:
                    streamed.input_transcript_parts.append(input_delta)
        except Exception as exc:
            append_jsonl(raw_events, {"receiver_error": f"{type(exc).__name__}: {exc}"})
        finally:
            receiver_done.set()

    receiver_task = asyncio.create_task(receiver())
    await ws.send(json.dumps(gemini_setup_payload(args)))
    append_jsonl(
        send_events,
        {
            "type": "setup",
            "sent_at_s": round(time.monotonic() - start_time, 6),
            "segment_index": segment_index,
        },
    )
    try:
        await asyncio.wait_for(setup_done.wait(), timeout=args.setup_timeout_s)
    except asyncio.TimeoutError:
        append_jsonl(raw_events, {"setup_timeout_s": args.setup_timeout_s})

    bytes_per_chunk = max(2, int(round(sample_rate * args.chunk_ms / 1000.0)) * 2)
    next_progress_s = args.progress_interval_s
    for send_index, start in enumerate(range(0, len(pcm), bytes_per_chunk)):
        payload = pcm[start : start + bytes_per_chunk]
        await ws.send(json.dumps(gemini_audio_payload(payload)))
        sent_audio_s = pcm16_duration_s(pcm[: start + len(payload)], sample_rate)
        elapsed_s = time.monotonic() - start_time
        append_jsonl(
            send_events,
            {
                "type": "realtime_input.audio",
                "send_index": send_index,
                "segment_index": segment_index,
                "sent_at_s": round(elapsed_s, 6),
                "sent_source_ms": round(
                    min(total_source_ms, base_source_ms + sent_audio_s * 1000.0),
                    3,
                ),
                "audio_duration_s": round(pcm16_duration_s(payload, sample_rate), 6),
                "bytes": len(payload),
            },
        )
        if args.progress_interval_s > 0 and elapsed_s >= next_progress_s:
            print_progress(
                args,
                out_dir,
                "sending",
                elapsed_s,
                base_source_ms / 1000.0 + sent_audio_s,
                len(streamed.delays_ms),
            )
            next_progress_s += args.progress_interval_s
        if args.pace:
            await asyncio.sleep(args.chunk_ms / 1000.0)

    send_finished_at = time.monotonic()
    await ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
    append_jsonl(
        send_events,
        {
            "type": "realtime_input.audio_stream_end",
            "sent_at_s": round(send_finished_at - start_time, 6),
            "segment_index": segment_index,
        },
    )
    while (
        not receiver_done.is_set()
        and time.monotonic() < send_finished_at + args.receive_timeout_s
    ):
        idle_s = time.monotonic() - max(last_event_time, send_finished_at)
        if idle_s >= args.post_send_idle_s:
            break
        await asyncio.sleep(0.1)

    await ws.close()
    receiver_task.cancel()
    return streamed


def print_progress(
    args: argparse.Namespace,
    out_dir: Path,
    phase: str,
    elapsed_s: float,
    sent_audio_s: float,
    output_units: int,
) -> None:
    print(
        json.dumps(
            {
                "run_id": out_dir.name,
                "provider": args.provider,
                "phase": phase,
                "elapsed_s": round(elapsed_s, 3),
                "sent_audio_s": round(sent_audio_s, 3),
                "output_units": output_units,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def pcm_segment_ranges(
    pcm: bytes,
    sample_rate: int,
    max_session_input_s: float,
) -> list[tuple[int, int]]:
    if max_session_input_s <= 0:
        return [(0, len(pcm))]
    max_bytes = max(2, int(round(max_session_input_s * sample_rate)) * 2)
    return [
        (start, min(len(pcm), start + max_bytes))
        for start in range(0, len(pcm), max_bytes)
    ]


async def run_stream(
    stream_wav: Path,
    api_key: str,
    out_dir: Path,
    args: argparse.Namespace,
    total_source_ms: float,
) -> StreamedText:
    sample_rate = OPENAI_SAMPLE_RATE if args.provider == "openai" else GEMINI_SAMPLE_RATE
    pcm = wav_to_pcm16(stream_wav)
    if args.dry_run:
        text = "这是一个流式输入烟测。"
        units = count_latency_units(text, args.target_lang)
        return StreamedText(
            text_parts=[text],
            delays_ms=[min(total_source_ms, args.chunk_ms)] * units,
            elapsed_ms=[min(total_source_ms, args.chunk_ms)] * units,
            input_transcript_parts=[],
            errors=[],
        )
    if args.provider == "openai":
        return await run_openai_stream(pcm, sample_rate, api_key, out_dir, args, total_source_ms)
    ranges = pcm_segment_ranges(pcm, sample_rate, args.max_session_input_s)
    if len(ranges) == 1:
        return await run_gemini_stream(pcm, sample_rate, api_key, out_dir, args, total_source_ms)

    combined = StreamedText([], [], [], [], [])
    run_start_time = time.monotonic()
    for segment_index, (start, end) in enumerate(ranges):
        base_source_ms = pcm16_duration_s(pcm[:start], sample_rate) * 1000.0
        part = await run_gemini_stream(
            pcm[start:end],
            sample_rate,
            api_key,
            out_dir,
            args,
            total_source_ms,
            base_source_ms=base_source_ms,
            segment_index=segment_index,
            run_start_time=run_start_time,
            reset_logs=segment_index == 0,
        )
        combined.text_parts.extend(part.text_parts)
        combined.delays_ms.extend(part.delays_ms)
        combined.elapsed_ms.extend(part.elapsed_ms)
        combined.input_transcript_parts.extend(part.input_transcript_parts)
        combined.errors.extend(part.errors)
    return combined


def select_indices(total: int, start_index: int, limit: int) -> list[int]:
    end = total if limit <= 0 else min(total, start_index + limit)
    return list(range(start_index, end))


async def async_main() -> None:
    args = parse_args()
    if args.download_hf:
        download_hf_subset(args.dataset_root)
    paths = resolve_paths(args.dataset_root, args.target_lang)
    source_lines = read_lines(paths.source_list)
    target_lines = read_lines(paths.target_list)
    audio_rows = parse_simple_audio_yaml(paths.audio_yaml)
    if len(source_lines) != len(target_lines):
        raise ValueError(
            f"source/target line count mismatch: {paths.source_list} vs {paths.target_list}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    instances_path = args.output_dir / "instances.log"
    responses_path = args.output_dir / "responses.jsonl"
    if not args.resume:
        instances_path.write_text("", encoding="utf-8")
        responses_path.write_text("", encoding="utf-8")
    done = load_done_indices(instances_path) if args.resume else set()
    api_key = read_secret(args.api_key_file)
    model = args.model or (OPENAI_MODEL if args.provider == "openai" else GEMINI_MODEL)
    run_config = {
        "provider": args.provider,
        "model": model,
        "dataset_root": str(args.dataset_root),
        "source_list": str(paths.source_list),
        "target_list": str(paths.target_list),
        "ref_file": str(paths.ref_file),
        "source_text_file": str(paths.source_text_file),
        "audio_yaml": str(paths.audio_yaml),
        "audio_yaml_rows": len(audio_rows),
        "chunk_ms": args.chunk_ms,
        "pace": args.pace,
        "max_audio_seconds": args.max_audio_seconds,
        "max_session_input_s": args.max_session_input_s,
        "target_lang": args.target_lang,
        "hf_repo": HF_REPO,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    sample_rate = OPENAI_SAMPLE_RATE if args.provider == "openai" else GEMINI_SAMPLE_RATE
    for index in select_indices(len(source_lines), args.start_index, args.limit):
        if index in done:
            print(json.dumps({"skip_done": index}, ensure_ascii=False), flush=True)
            continue
        source_wav = resolve_audio_path(source_lines[index], args.dataset_root, paths)
        run_id = source_wav.stem
        run_dir = args.output_dir / f"{index:03d}_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        stream_wav = run_dir / f"source_stream_{sample_rate}.wav"
        ffmpeg_convert(source_wav, stream_wav, sample_rate, args.max_audio_seconds)
        source_length_ms = wav_duration_ms(stream_wav if args.max_audio_seconds > 0 else source_wav)
        start_s = time.monotonic()
        streamed = await run_stream(stream_wav, api_key, run_dir, args, source_length_ms)
        elapsed_ms = (time.monotonic() - start_s) * 1000.0
        prediction = "".join(streamed.text_parts).strip()
        if not streamed.delays_ms:
            fallback_units = max(1, count_latency_units(prediction, args.target_lang))
            streamed.delays_ms = [source_length_ms] * fallback_units
            streamed.elapsed_ms = [round(elapsed_ms, 3)] * fallback_units
        record = {
            "index": index,
            "prediction": prediction,
            "delays": streamed.delays_ms,
            "elapsed": streamed.elapsed_ms,
            "prediction_length": len(streamed.delays_ms),
            "reference": target_lines[index],
            "source": [str(source_wav)],
            "source_length": source_length_ms,
        }
        append_jsonl(instances_path, record)
        response = {
            "index": index,
            "run_id": run_id,
            "provider": args.provider,
            "model": model,
            "prediction": prediction,
            "prediction_units": len(streamed.delays_ms),
            "elapsed_ms": round(elapsed_ms, 3),
            "input_transcript": "".join(streamed.input_transcript_parts).strip(),
            "error_count": len(streamed.errors),
            "run_dir": str(run_dir),
        }
        append_jsonl(responses_path, response)
        print(
            json.dumps(
                {
                    "index": index,
                    "run_id": run_id,
                    "prediction_units": len(streamed.delays_ms),
                    "error_count": len(streamed.errors),
                    "elapsed_ms": round(elapsed_ms, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def main() -> None:
    try:
        asyncio.run(async_main())
    except ModuleNotFoundError as exc:
        if exc.name == "websockets":
            raise SystemExit("websockets is required for live streaming runs") from exc
        raise


if __name__ == "__main__":
    main()
