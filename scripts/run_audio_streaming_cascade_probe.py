#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl, write_jsonl
from s2s_omni.metrics import unit_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run speech-chunk -> Qwen3-Omni thinker -> Higgs TTS streaming cascade."
    )
    parser.add_argument("--input", required=True, help="Stream manifest JSONL from build_gigaspeech_audio_chunk_manifest.py.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--source-lang", default="English")
    parser.add_argument("--target-lang", default="Chinese")
    parser.add_argument("--system-prompt-style", choices=["translate_task", "given_chunks"], default="translate_task")
    parser.add_argument("--max-cache-chunks", type=int, default=120)
    parser.add_argument("--keep-cache-chunks", type=int, default=60)
    parser.add_argument("--max-streams", type=int, default=0)
    parser.add_argument("--max-chunks-per-stream", type=int, default=0)
    parser.add_argument("--warmup-audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tts-url", default="http://127.0.0.1:18111/v1/audio/speech")
    parser.add_argument("--tts-timeout-s", type=float, default=300.0)
    parser.add_argument("--tts-language", default="Chinese")
    parser.add_argument("--tts-frame-rate", type=float, default=25.0)
    parser.add_argument("--tts-token-budget-margin", type=int, default=0)
    parser.add_argument("--tts-ref-mode", choices=["none", "chunk", "stream_first"], default="none")
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def clean_completion(text: str) -> str:
    text = (text or "").strip()
    for token in ["<|im_end|>", "<|endoftext|>"]:
        text = text.replace(token, "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")
    for prefix in ["Translation:", "Answer:", "译文：", "译文:"]:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return " ".join(text.strip().split())


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180] or "sample"


def load_wav(path: str) -> tuple[np.ndarray, int, float]:
    import soundfile as sf

    started = time.perf_counter()
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    load_s = time.perf_counter() - started
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    if array.ndim != 1:
        array = array.reshape(-1)
    return np.nan_to_num(array), int(sample_rate), load_s


def wav_duration_s(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate:
            return round(float(info.frames) / float(info.samplerate), 6)
    except Exception:
        return None
    return None


def system_prompt(style: str, source_lang: str, target_lang: str) -> str:
    if style == "given_chunks":
        return (
            "You are a professional simultaneous interpreter. "
            f"You will be given chunks of {source_lang} audio and you need to "
            f"translate the audio into {target_lang} text."
        )
    return (
        "You are a professional simultaneous interpreter. "
        f"Your task is to translate {source_lang} audio chunks into accurate and fluent {target_lang}."
    )


class StreamingThinker:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoConfig, Qwen3OmniMoeProcessor
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration

        self.torch = torch
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(args.model, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        full_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        kwargs: dict[str, Any] = {
            "config": full_config.thinker_config,
            "trust_remote_code": True,
            "dtype": "auto",
            "device_map": args.device_map,
        }
        if args.attn_implementation:
            kwargs["attn_implementation"] = args.attn_implementation
        self.model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(args.model, **kwargs).eval()
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k

    def generate(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        preprocess_started = time.perf_counter()
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios = [
            item["audio"]
            for message in messages
            for item in message.get("content", [])
            if isinstance(item, dict) and item.get("type") == "audio"
        ]
        inputs = self.processor(
            text=text,
            audio=audios,
            images=None,
            videos=None,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                if key == "input_features":
                    inputs[key] = value.to(device=device, dtype=model_dtype)
                else:
                    inputs[key] = value.to(device=device)
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        preprocess_s = time.perf_counter() - preprocess_started

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p
            if self.top_k > 0:
                generate_kwargs["top_k"] = self.top_k
        generate_started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, **generate_kwargs)
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        generate_s = time.perf_counter() - generate_started
        new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
        completion = clean_completion(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return {
            "text": completion,
            "thinker_preprocess_s": round(preprocess_s, 6),
            "thinker_generate_s": round(generate_s, 6),
            "thinker_wall_s": round(preprocess_s + generate_s, 6),
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "generated_tokens": int(new_tokens.shape[0]),
            "audio_count_in_prompt": len(audios),
        }


class HiggsClient:
    def __init__(self, args: argparse.Namespace) -> None:
        self.url = args.tts_url
        self.timeout_s = args.tts_timeout_s
        self.language = args.tts_language
        self.frame_rate = args.tts_frame_rate
        self.token_budget_margin = args.tts_token_budget_margin
        self.ref_mode = args.tts_ref_mode

    def synthesize(
        self,
        text: str,
        output_wav: Path,
        chunk_duration_s: float,
        chunk_audio_path: str,
        first_audio_path: str | None,
    ) -> dict[str, Any]:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        token_budget = max(1, int(math.ceil(chunk_duration_s * self.frame_rate)) + self.token_budget_margin)
        payload: dict[str, Any] = {
            "input": text,
            "language": self.language,
            "max_new_tokens": token_budget,
            "token_count": token_budget,
        }
        ref_audio = None
        if self.ref_mode == "chunk":
            ref_audio = chunk_audio_path
        elif self.ref_mode == "stream_first":
            ref_audio = first_audio_path
        if ref_audio:
            payload["references"] = [{"audio_path": ref_audio, "text": ""}]
        started = time.perf_counter()
        if text.strip():
            response = requests.post(self.url, json=payload, timeout=self.timeout_s)
            status_code = response.status_code
            content_type = response.headers.get("content-type", "")
            error_text = ""
            if response.ok:
                output_wav.write_bytes(response.content)
            else:
                error_text = response.text[-2000:]
        else:
            status_code = 204
            content_type = ""
            error_text = ""
        wall_s = time.perf_counter() - started
        return {
            "higgs_wall_s": round(wall_s, 6),
            "higgs_status_code": status_code,
            "higgs_content_type": content_type,
            "higgs_error_tail": error_text,
            "higgs_token_budget": token_budget,
            "higgs_ref_mode": self.ref_mode,
        }


def prune_messages(messages: list[dict[str, Any]], max_cache_chunks: int, keep_cache_chunks: int) -> list[dict[str, Any]]:
    body = messages[1:]
    if max_cache_chunks <= 0 or len(body) <= 2 * max_cache_chunks:
        return messages
    return [messages[0]] + body[-2 * keep_cache_chunks :]


def stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    values = sorted(values)
    return {
        "min": round(values[0], 6),
        "p50": round(values[len(values) // 2], 6),
        "p90": round(values[min(len(values) - 1, int(len(values) * 0.9))], 6),
        "max": round(values[-1], 6),
        "mean": round(sum(values) / len(values), 6),
    }


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    numeric_keys = [
        "chunk_duration_s",
        "audio_load_s",
        "thinker_preprocess_s",
        "thinker_generate_s",
        "thinker_wall_s",
        "higgs_wall_s",
        "playback_duration_s",
        "cascade_compute_s",
        "serial_total_s",
        "deadline_ratio_compute",
        "deadline_ratio_with_playback",
        "compute_backlog_after_s",
        "playback_backlog_after_s",
    ]
    out: dict[str, Any] = {
        "records": len(rows),
        "streams": len({row["stream_id"] for row in rows}),
        "model": args.model,
        "tts_url": args.tts_url,
        "tts_ref_mode": args.tts_ref_mode,
        "system_prompt_style": args.system_prompt_style,
        "deadline_miss_compute": sum(1 for row in rows if not row.get("compute_fits_chunk")),
        "deadline_miss_with_playback": sum(1 for row in rows if not row.get("serial_fits_chunk")),
        "higgs_success": sum(1 for row in rows if int(row.get("higgs_status_code", 0)) in {200, 204}),
    }
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        out[key] = stats(values)
    return out


def run_stream(
    stream: dict[str, Any],
    thinker: StreamingThinker,
    tts: HiggsClient,
    args: argparse.Namespace,
    output_audio_dir: Path,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt(args.system_prompt_style, args.source_lang, args.target_lang)}],
        }
    ]
    rows: list[dict[str, Any]] = []
    compute_backlog = 0.0
    playback_backlog = 0.0
    chunks = stream.get("chunks", [])
    if args.max_chunks_per_stream > 0:
        chunks = chunks[: args.max_chunks_per_stream]
    first_audio_path = chunks[0]["audio_path"] if chunks else None

    for chunk_index, chunk in enumerate(chunks):
        audio, sample_rate, audio_load_s = load_wav(chunk["audio_path"])
        user_message = {"role": "user", "content": [{"type": "audio", "audio": audio}]}
        messages.append(user_message)
        thinker_out = thinker.generate(messages)
        target_text = thinker_out["text"]
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target_text}]})
        messages = prune_messages(messages, args.max_cache_chunks, args.keep_cache_chunks)

        duration_s = float(chunk.get("duration_s") or chunk.get("end_s", 0) - chunk.get("start_s", 0))
        output_wav = output_audio_dir / safe_name(stream["id"]) / f"{chunk_index:04d}_{safe_name(chunk['id'])}.wav"
        higgs_out = tts.synthesize(target_text, output_wav, duration_s, chunk["audio_path"], first_audio_path)
        playback_s = wav_duration_s(output_wav)
        cascade_compute_s = thinker_out["thinker_wall_s"] + higgs_out["higgs_wall_s"]
        serial_total_s = cascade_compute_s + (playback_s or 0.0)
        compute_backlog = max(0.0, compute_backlog + cascade_compute_s - duration_s)
        playback_backlog = max(0.0, playback_backlog + serial_total_s - duration_s)
        row = {
            "stream_id": stream["id"],
            "chunk_id": chunk["id"],
            "chunk_index": chunk_index,
            "chunk_start_s": chunk.get("start_s"),
            "chunk_end_s": chunk.get("end_s"),
            "chunk_duration_s": round(duration_s, 6),
            "chunk_audio_path": chunk["audio_path"],
            "source_sample_rate": sample_rate,
            "audio_load_s": round(audio_load_s, 6),
            "target_text": target_text,
            "target_units": unit_count(target_text, stream.get("tgt_lang", "zh")),
            "output_wav": str(output_wav),
            "playback_duration_s": playback_s,
            "cascade_compute_s": round(cascade_compute_s, 6),
            "serial_total_s": round(serial_total_s, 6),
            "deadline_ratio_compute": round(cascade_compute_s / duration_s, 6),
            "deadline_ratio_with_playback": round(serial_total_s / duration_s, 6),
            "compute_fits_chunk": cascade_compute_s <= duration_s,
            "serial_fits_chunk": serial_total_s <= duration_s,
            "compute_backlog_after_s": round(compute_backlog, 6),
            "playback_backlog_after_s": round(playback_backlog, 6),
            **thinker_out,
            **higgs_out,
        }
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    streams = list(read_jsonl(args.input))
    if args.max_streams > 0:
        streams = streams[: args.max_streams]
    thinker = StreamingThinker(args)
    if args.warmup_audio:
        warmup_messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt(args.system_prompt_style, args.source_lang, args.target_lang)}],
            },
            {"role": "user", "content": [{"type": "audio", "audio": np.zeros(16000, dtype=np.float32)}]},
        ]
        thinker.generate(warmup_messages)
    tts = HiggsClient(args)
    output_audio_dir = Path(args.audio_dir)
    rows: list[dict[str, Any]] = []
    for stream in streams:
        rows.extend(run_stream(stream, thinker, tts, args, output_audio_dir))
        if args.log_every > 0 and rows and len(rows) % args.log_every == 0:
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    write_jsonl(args.output, rows)
    summary = summarize(rows, args)
    summary_path = Path(args.summary_output or str(Path(args.output).with_suffix(".summary.json")))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
