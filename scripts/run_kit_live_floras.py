#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream a wav file into KIT Lecture Translator.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--wav-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--cookie-header-file", required=True)
    parser.add_argument("--base-url", default="https://lecture-translator.kit.edu")
    parser.add_argument("--chunk-s", type=float, default=1.0)
    parser.add_argument("--pace", choices=["realtime", "accelerated"], default="realtime")
    parser.add_argument("--accelerated-sleep-s", type=float, default=0.12)
    parser.add_argument("--poll-after-s", type=float, default=120.0)
    parser.add_argument("--poll-interval-s", type=float, default=15.0)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_cookie_header(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def request_json_or_text(
    *,
    url: str,
    cookie_header: str,
    payload: Any | None = None,
    timeout_s: float = 30.0,
    retries: int = 5,
    retry_sleep_s: float = 3.0,
) -> tuple[int, bool, str]:
    data = None
    headers = {"Cookie": cookie_header}
    if payload is not None:
        data = json.dumps(json.dumps(payload)).encode("utf-8")
        headers.update({"Accept": "application/json", "Content-Type": "application/json"})
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    last_error = ""
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                text = response.read().decode("utf-8", errors="replace")
                return int(response.status), 200 <= int(response.status) < 300, text
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            return int(exc.code), False, text
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = repr(exc)
            if attempt >= retries:
                break
            time.sleep(retry_sleep_s)
    return 0, False, last_error


def load_pcm16le_mono_16k(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    if sample_width != 2:
        raise ValueError(f"{path} must be 16-bit PCM, got sample width {sample_width}")
    if channels != 1 or sample_rate != 16000:
        # audioop was removed in Python 3.13; only needed for downmix/resample.
        try:
            import audioop
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                f"{path} is {channels}ch/{sample_rate}Hz and needs conversion, but the "
                "'audioop' module is unavailable (Python >= 3.13). Provide a 16kHz mono "
                "16-bit PCM wav, or run under Python <= 3.12."
            ) from exc
        if channels != 1:
            raw = audioop.tomono(raw, sample_width, 0.5, 0.5)
        if sample_rate != 16000:
            raw, _ = audioop.ratecv(raw, sample_width, 1, sample_rate, 16000, None)
    return raw, len(raw) / 2 / 16000.0


def post_audio_chunk(
    *,
    base_url: str,
    session_id: str,
    cookie_header: str,
    pcm_chunk: bytes,
    start_s: float,
) -> dict[str, Any]:
    payload = {
        "b64_enc_pcm_s16le": base64.b64encode(pcm_chunk).decode("ascii"),
        "start": start_s,
    }
    sent_at = time.monotonic()
    status, ok, text = request_json_or_text(
        url=f"{base_url}/webapi/{session_id}/0/append",
        cookie_header=cookie_header,
        payload=payload,
        timeout_s=60.0,
    )
    return {
        "offset": int(start_s * 32000),
        "bytes": len(pcm_chunk),
        "audio_end_s": round(start_s + len(pcm_chunk) / 32000.0, 3),
        "status": status,
        "ok": ok,
        "ms": round((time.monotonic() - sent_at) * 1000.0),
        "text_start": text[:200],
    }


def post_pause(*, base_url: str, session_id: str, cookie_header: str) -> dict[str, Any]:
    payload = {"b64_enc_pcm_s16le": base64.b64encode(b"PAUSE").decode("ascii")}
    status, ok, text = request_json_or_text(
        url=f"{base_url}/webapi/{session_id}/0/append",
        cookie_header=cookie_header,
        payload=payload,
        timeout_s=60.0,
    )
    return {"status": status, "ok": ok, "text_start": text[:200], "at": now_iso()}


def collect_messages(*, base_url: str, session_id: str, cookie_header: str) -> dict[str, Any]:
    graph_status, graph_ok, graph_text = request_json_or_text(
        url=f"{base_url}/webapi/{session_id}/getgraph",
        cookie_header=cookie_header,
        payload=None,
        timeout_s=30.0,
    )
    graph = json.loads(graph_text) if graph_ok and graph_text.strip() else {}
    sender_map: dict[str, str] = {}
    messages_by_component: dict[str, list[dict[str, Any]]] = {}
    for worker in sorted(graph):
        values = graph.get(worker)
        if not isinstance(values, list) or "api" not in values:
            continue
        stream = worker.split(":")[1]
        component_payload = {"component": worker}
        status, ok, text = request_json_or_text(
            url=f"{base_url}/webapi/{session_id}/{stream}/get_output_language_component",
            cookie_header=cookie_header,
            payload=component_payload,
            timeout_s=30.0,
        )
        if ok:
            sender_map[worker] = text
        status, ok, text = request_json_or_text(
            url=f"{base_url}/webapi/{session_id}/get_previous_messages",
            cookie_header=cookie_header,
            payload={"component": worker, "begin": 0},
            timeout_s=60.0,
        )
        if ok and text.strip():
            parsed = json.loads(text)
            messages_by_component[worker] = [json.loads(item) for item in parsed]
        else:
            messages_by_component[worker] = []
    return {
        "graphStatus": graph_status,
        "graph": graph,
        "senderMap": sender_map,
        "messagesByComponent": messages_by_component,
    }


def collection_counts(collection: dict[str, Any]) -> dict[str, int]:
    return {
        key: len(value)
        for key, value in collection.get("messagesByComponent", {}).items()
        if isinstance(value, list)
    }


def write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_json).expanduser()
    wav_path = Path(args.wav_path).expanduser()
    cookie_header = read_cookie_header(Path(args.cookie_header_file).expanduser())
    pcm, duration_s = load_pcm16le_mono_16k(wav_path)
    bytes_per_second = 32000
    chunk_bytes = int(round(args.chunk_s * bytes_per_second))
    if chunk_bytes <= 0:
        raise ValueError("--chunk-s must be positive")
    if chunk_bytes % 2:
        chunk_bytes += 1
    result: dict[str, Any] = {
        "sessionId": args.session_id,
        "sessionUrl": f"{args.base_url}/present/{args.session_id}",
        "sessionName": args.session_name,
        "wavPath": str(wav_path),
        "pacing": args.pace,
        "chunkS": args.chunk_s,
        "sourceDurationS": duration_s,
        "startedAt": now_iso(),
        "postStats": [],
        "pollStats": [],
    }
    write_output(output_path, result)

    chunks = [(offset, pcm[offset : offset + chunk_bytes]) for offset in range(0, len(pcm), chunk_bytes)]
    start_wall = time.monotonic()
    for index, (offset, chunk) in enumerate(chunks, start=1):
        audio_start_s = offset / bytes_per_second
        stat = post_audio_chunk(
            base_url=args.base_url,
            session_id=args.session_id,
            cookie_header=cookie_header,
            pcm_chunk=chunk,
            start_s=audio_start_s,
        )
        result["postStats"].append(stat)
        if index == 1 or index == len(chunks) or index % 30 == 0:
            collection = collect_messages(
                base_url=args.base_url,
                session_id=args.session_id,
                cookie_header=cookie_header,
            )
            result["pollStats"].append(
                {"at": now_iso(), "audioEndS": stat["audio_end_s"], "counts": collection_counts(collection)}
            )
            result["collection"] = collection
            write_output(output_path, result)
        if not stat["ok"]:
            result["error"] = f"audio post failed at chunk {index}: {stat}"
            write_output(output_path, result)
            raise RuntimeError(result["error"])
        if args.pace == "realtime":
            target_elapsed = audio_start_s + len(chunk) / bytes_per_second
            sleep_s = target_elapsed - (time.monotonic() - start_wall)
        else:
            sleep_s = args.accelerated_sleep_s
        if sleep_s > 0:
            time.sleep(sleep_s)

    result["pauseStatus"] = post_pause(
        base_url=args.base_url,
        session_id=args.session_id,
        cookie_header=cookie_header,
    )
    result["finishedSendingAt"] = now_iso()
    write_output(output_path, result)

    poll_deadline = time.monotonic() + args.poll_after_s
    last_counts: dict[str, int] | None = None
    stable_polls = 0
    while time.monotonic() < poll_deadline:
        collection = collect_messages(
            base_url=args.base_url,
            session_id=args.session_id,
            cookie_header=cookie_header,
        )
        counts = collection_counts(collection)
        result["pollStats"].append({"at": now_iso(), "audioEndS": duration_s, "counts": counts})
        result["collection"] = collection
        write_output(output_path, result)
        if counts == last_counts:
            stable_polls += 1
            if stable_polls >= 3:
                break
        else:
            stable_polls = 0
            last_counts = counts
        time.sleep(args.poll_interval_s)

    result["finishedAt"] = now_iso()
    write_output(output_path, result)
    print(json.dumps({"output": str(output_path), "duration_s": duration_s, "counts": last_counts}, indent=2))


if __name__ == "__main__":
    main()
