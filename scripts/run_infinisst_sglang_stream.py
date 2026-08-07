#!/usr/bin/env python3
"""Chunked streaming InfiniSST S2T against an sglang Qwen3-Omni server.

Replicates the RASST no-RAG batched eval protocol client-side:
system interpreter prompt, per-chunk user message carrying only the new
audio increment (base64 wav) plus ``term_map:\nNONE``, assistant text
appended to the dialogue, sliding message cache (max 16 / keep 8 chunks),
sampling T=0.6 top_p=0.95 top_k=20, max_new_tokens 40.

Outputs one runtime JSONL per talk with llm_output-style records
(segment_idx, text, delay_ms) consumable by the v2 TTS stage.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.request
import wave
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", required=True, help="16k mono source wav")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:47210")
    parser.add_argument("--model", default="infinisst")
    parser.add_argument("--source-lang", default="English")
    parser.add_argument("--target-lang", default="Chinese")
    parser.add_argument("--chunk-s", type=float, default=0.96)
    parser.add_argument("--max-cache-chunks", type=int, default=16)
    parser.add_argument("--keep-cache-chunks", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def wav_b64(pcm: bytes, rate: int) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    args = parse_args()
    with wave.open(args.wav, "rb") as w:
        rate = w.getframerate()
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, "need 16k mono s16le"
        pcm = w.readframes(w.getnframes())
    chunk_bytes = int(args.chunk_s * rate) * 2
    total_chunks = (len(pcm) + chunk_bytes - 1) // chunk_bytes

    system_text = (
        f"You are a professional simultaneous interpreter. "
        f"Your task is to translate {args.source_lang} audio chunks into accurate "
        f"and fluent {args.target_lang}. Use the 'term_map' as a reference for "
        f"terminology if provided."
    )
    messages: list[dict] = [{"role": "system", "content": system_text}]

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"

    with out_path.open("w", encoding="utf-8") as out:
        for idx in range(total_chunks):
            increment = pcm[idx * chunk_bytes : (idx + 1) * chunk_bytes]
            user_msg = {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "data:audio/wav;base64," + wav_b64(increment, rate)},
                    },
                    {"type": "text", "text": "\n\nterm_map:\nNONE"},
                ],
            }
            messages.append(user_msg)
            payload = {
                "model": args.model,
                "messages": messages,
                "max_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "extra_body": {"top_k": args.top_k},
            }
            started = time.perf_counter()
            req = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=args.timeout_s) as resp:
                body = json.loads(resp.read())
            text = (body["choices"][0]["message"]["content"] or "").strip()
            messages.append({"role": "assistant", "content": text})

            # sliding cache: keep system + last 2*keep_cache_chunks messages
            if len(messages) >= 2 * args.max_cache_chunks + 1:
                messages = [messages[0]] + messages[-2 * args.keep_cache_chunks :]

            record = {
                "type": "llm_output",
                "segment_idx": idx,
                "text": text,
                "delay_ms": round((idx + 1) * args.chunk_s * 1000.0, 3),
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "source_path": args.wav,
                "chunk_s": args.chunk_s,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if args.log_every > 0 and (idx + 1) % args.log_every == 0:
                print(
                    json.dumps({"chunk": idx + 1, "total": total_chunks}),
                    flush=True,
                )

    print(json.dumps({"wav": args.wav, "chunks": total_chunks, "done": True}))


if __name__ == "__main__":
    main()
