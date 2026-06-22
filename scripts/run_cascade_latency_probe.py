#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl, write_jsonl
from s2s_omni.metrics import unit_count
from s2s_omni.prompts import SYSTEM_COMPRESSION, build_compression_user_prompt
from s2s_omni.schema import S2SSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe cascade S2ST latency: thinker text, TTS wall time, playback, and backlog."
    )
    parser.add_argument("--input", required=True, help="Sample/prediction JSONL.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=260622)
    parser.add_argument(
        "--text-source",
        choices=["record", "compressed", "reference", "evo_prediction", "base_prediction", "thinker"],
        default="record",
    )
    parser.add_argument("--thinker-backend", choices=["none", "transformers", "openai"], default="none")
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--adapter")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tts-backend", choices=["command", "qwen3_tts"], default="command")
    parser.add_argument("--tts-command", default="")
    parser.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--tts-language", default="Chinese")
    parser.add_argument("--tts-device-map", default="cuda:0")
    parser.add_argument("--tts-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--tts-attn-implementation", default="sdpa")
    parser.add_argument("--tts-ref-audio", default="")
    parser.add_argument("--tts-ref-text", default="")
    parser.add_argument("--tts-x-vector-only-mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tts-max-new-tokens", type=int, default=2048)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--source-duration-key", default="")
    parser.add_argument("--target-sample-rate", type=int, default=24000)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def load_records(path: str, num_samples: int, seed: int) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    if num_samples > 0 and len(rows) > num_samples:
        rng = random.Random(seed)
        rows = rng.sample(rows, num_samples)
    return rows


def sample_from_record(record: dict[str, Any]) -> S2SSample:
    if isinstance(record.get("sample"), dict):
        payload = record["sample"]
    else:
        payload = dict(record)
        payload.setdefault("src_lang", "en")
        payload.setdefault("tgt_lang", "zh")
        if "prediction" not in payload and payload.get("evo_prediction"):
            payload["prediction"] = payload["evo_prediction"]
    return S2SSample.from_dict(payload)


def clean_completion(text: str) -> str:
    text = (text or "").strip()
    for token in ["<|im_end|>", "<|endoftext|>"]:
        text = text.replace(token, "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")
    text = text.strip().strip("`").strip()
    for prefix in [
        "Compressed translation:",
        "Translation:",
        "Answer:",
        "译文：",
        "译文:",
        "压缩译文：",
        "压缩译文:",
    ]:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return " ".join(text.split())


def choose_text(record: dict[str, Any], sample: S2SSample, text_source: str) -> str:
    if text_source == "compressed":
        return sample.compressed_translation or record.get("compressed_translation") or ""
    if text_source == "reference":
        return sample.reference_translation or ""
    if text_source == "evo_prediction":
        return record.get("evo_prediction") or record.get("prediction") or ""
    if text_source == "base_prediction":
        return record.get("base_prediction") or ""
    return (
        record.get("target_text")
        or record.get("compressed_translation")
        or sample.compressed_translation
        or record.get("evo_prediction")
        or sample.reference_translation
        or ""
    )


def source_budget_s(sample: S2SSample, record: dict[str, Any], key: str, text_source: str) -> float:
    if key and record.get(key) is not None:
        return max(1.0e-6, float(record[key]))
    prefixes = {
        "base_prediction": ["base"],
        "evo_prediction": ["evo"],
        "record": ["evo", "base"],
        "compressed": ["evo", "base"],
        "reference": ["evo", "base"],
        "thinker": ["evo", "base"],
    }.get(text_source, ["evo", "base"])
    for prefix in prefixes:
        estimated = record.get(f"{prefix}_estimated_target_speech_s")
        rtf = record.get(f"{prefix}_s2s_rtf")
        if estimated and rtf:
            return max(1.0e-6, float(estimated) / float(rtf))
    metadata = sample.metadata or {}
    for candidate in [
        metadata.get("source_wall_duration_s"),
        metadata.get("playback_budget_s"),
        sample.target.max_target_duration_s,
        sample.timing.src_duration_s,
    ]:
        if candidate:
            return max(1.0e-6, float(candidate))
    return 1.0e-6


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


def torch_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


class Thinker:
    def generate(self, sample: S2SSample) -> str:
        raise NotImplementedError


class NullThinker(Thinker):
    def generate(self, sample: S2SSample) -> str:
        raise RuntimeError("NullThinker cannot generate text")


class OpenAIThinker(Thinker):
    def __init__(self, args: argparse.Namespace) -> None:
        from s2s_omni.llm_client import ChatClient

        self.client = ChatClient(
            base_url=args.base_url.rstrip("/"),
            api_key=args.api_key,
            model=args.model,
            timeout_s=args.timeout_s,
        )
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature

    def generate(self, sample: S2SSample) -> str:
        return self.client.chat(
            messages=messages_for_sample(sample),
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )


class TransformersThinker(Thinker):
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
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            args.model,
            config=full_config.thinker_config,
            trust_remote_code=True,
            dtype="auto",
            device_map=args.device_map,
        )
        if args.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter)
        self.model = model.eval()
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature

    def generate(self, sample: S2SSample) -> str:
        prompt = self.processor.apply_chat_template(
            messages_for_sample(sample),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, **kwargs)
        new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
        return clean_completion(self.tokenizer.decode(new_tokens, skip_special_tokens=True))


def messages_for_sample(sample: S2SSample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_COMPRESSION.strip()},
        {"role": "user", "content": build_compression_user_prompt(sample, include_reference=False)},
    ]


def make_thinker(args: argparse.Namespace) -> Thinker:
    if args.thinker_backend == "transformers":
        return TransformersThinker(args)
    if args.thinker_backend == "openai":
        return OpenAIThinker(args)
    return NullThinker()


class TTSRunner:
    def synthesize(self, sample: S2SSample, text: str, output_wav: Path) -> dict[str, Any]:
        raise NotImplementedError


class CommandTTSRunner(TTSRunner):
    def __init__(self, command: str, timeout_s: float) -> None:
        if not command:
            raise ValueError("--tts-backend command requires --tts-command")
        self.command = command
        self.timeout_s = timeout_s

    def synthesize(self, sample: S2SSample, text: str, output_wav: Path) -> dict[str, Any]:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "S2S_SAMPLE_ID": sample.id,
                "S2S_TEXT": text,
                "S2S_OUTPUT_WAV": str(output_wav),
                "S2S_SOURCE_AUDIO": sample.audio_path or "",
                "S2S_SOURCE_TEXT": sample.source_text or "",
                "S2S_REFERENCE_TEXT": sample.reference_translation or "",
                "S2S_TGT_LANG": sample.tgt_lang,
            }
        )
        started = time.perf_counter()
        proc = subprocess.run(
            self.command,
            shell=True,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_s,
            check=False,
        )
        wall_s = time.perf_counter() - started
        return {
            "tts_backend": "command",
            "tts_wall_s": round(wall_s, 6),
            "tts_returncode": proc.returncode,
            "tts_stdout_tail": proc.stdout[-2000:],
            "tts_stderr_tail": proc.stderr[-2000:],
        }


class Qwen3TTSRunner(TTSRunner):
    def __init__(self, args: argparse.Namespace) -> None:
        import torch

        sys.modules.setdefault("kernels", None)
        from qwen_tts import Qwen3TTSModel

        started = time.perf_counter()
        self.model = Qwen3TTSModel.from_pretrained(
            args.tts_model,
            device_map=args.tts_device_map,
            dtype=torch_dtype(args.tts_dtype),
            attn_implementation=args.tts_attn_implementation,
        )
        self.load_s = round(time.perf_counter() - started, 6)
        self.torch = torch
        self.language = args.tts_language
        self.ref_audio = args.tts_ref_audio
        self.ref_text = args.tts_ref_text
        self.x_vector_only_mode = args.tts_x_vector_only_mode
        self.max_new_tokens = args.tts_max_new_tokens

    def synthesize(self, sample: S2SSample, text: str, output_wav: Path) -> dict[str, Any]:
        import soundfile as sf

        ref_audio, ref_text = self._reference(sample)
        started = time.perf_counter()
        with self.torch.inference_mode():
            if self.x_vector_only_mode:
                prompt = self.model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_text or "",
                    x_vector_only_mode=True,
                )
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language=self.language,
                    voice_clone_prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                )
            else:
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language=self.language,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    max_new_tokens=self.max_new_tokens,
                )
        wall_s = time.perf_counter() - started
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav), wavs[0], sr)
        return {
            "tts_backend": "qwen3_tts",
            "tts_model_load_s": self.load_s,
            "tts_wall_s": round(wall_s, 6),
            "tts_returncode": 0,
            "tts_stdout_tail": "",
            "tts_stderr_tail": "",
        }

    def _reference(self, sample: S2SSample) -> tuple[Any, str]:
        if self.ref_audio:
            return self.ref_audio, self.ref_text
        if not sample.audio_path:
            raise ValueError(f"{sample.id}: no source audio for Qwen3-TTS voice clone")
        from s2s_omni.audio import load_audio_span

        wav, sr = load_audio_span(sample.audio_path)
        return (wav, sr), self.ref_text or sample.source_text or ""


def make_tts_runner(args: argparse.Namespace) -> TTSRunner:
    if args.tts_backend == "qwen3_tts":
        return Qwen3TTSRunner(args)
    return CommandTTSRunner(args.tts_command, args.timeout_s)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180] or "sample"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "source_budget_s",
        "thinker_wall_s",
        "tts_wall_s",
        "playback_duration_s",
        "serial_total_s",
        "streaming_overlap_total_s",
        "serial_budget_ratio",
        "streaming_overlap_budget_ratio",
        "serial_backlog_after_s",
        "streaming_backlog_after_s",
    ]
    out: dict[str, Any] = {
        "records": len(rows),
        "tts_success": sum(1 for row in rows if row.get("tts_returncode") == 0 and row.get("playback_duration_s")),
        "serial_fits": sum(1 for row in rows if row.get("serial_fits_next_chunk")),
        "streaming_overlap_fits": sum(1 for row in rows if row.get("streaming_overlap_fits_next_chunk")),
    }
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        out[key] = stats(values)
    return out


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


def main() -> None:
    args = parse_args()
    if args.text_source == "thinker" and args.thinker_backend == "none":
        raise ValueError("--text-source thinker requires --thinker-backend transformers or openai")
    rows = load_records(args.input, args.num_samples, args.seed)
    thinker = make_thinker(args) if args.text_source == "thinker" else None
    tts_runner = make_tts_runner(args)
    output_rows: list[dict[str, Any]] = []
    serial_backlog = 0.0
    streaming_backlog = 0.0
    audio_dir = Path(args.audio_dir)

    for idx, record in enumerate(rows, start=1):
        sample = sample_from_record(record)
        budget_s = source_budget_s(sample, record, args.source_duration_key, args.text_source)
        thinker_wall_s = 0.0
        if args.text_source == "thinker":
            started = time.perf_counter()
            assert thinker is not None
            target_text = thinker.generate(sample)
            thinker_wall_s = time.perf_counter() - started
        else:
            target_text = str(choose_text(record, sample, args.text_source) or "")
        target_text = clean_completion(target_text)

        output_wav = audio_dir / f"{idx:04d}_{safe_name(sample.id)}.wav"
        tts = tts_runner.synthesize(sample, target_text, output_wav)
        playback_s = wav_duration_s(output_wav)
        serial_total = thinker_wall_s + tts["tts_wall_s"] + (playback_s or 0.0)
        streaming_total = thinker_wall_s + max(tts["tts_wall_s"], playback_s or 0.0)
        serial_backlog = max(0.0, serial_backlog + serial_total - budget_s)
        streaming_backlog = max(0.0, streaming_backlog + streaming_total - budget_s)
        row = {
            "id": sample.id,
            "source_text": sample.source_text,
            "target_text": target_text,
            "target_units": unit_count(target_text, sample.tgt_lang),
            "source_budget_s": round(budget_s, 6),
            "thinker_wall_s": round(thinker_wall_s, 6),
            "playback_duration_s": playback_s,
            "serial_total_s": round(serial_total, 6),
            "streaming_overlap_total_s": round(streaming_total, 6),
            "serial_budget_ratio": round(serial_total / budget_s, 6),
            "streaming_overlap_budget_ratio": round(streaming_total / budget_s, 6),
            "serial_fits_next_chunk": serial_total <= budget_s,
            "streaming_overlap_fits_next_chunk": streaming_total <= budget_s,
            "serial_backlog_after_s": round(serial_backlog, 6),
            "streaming_backlog_after_s": round(streaming_backlog, 6),
            "output_wav": str(output_wav),
            **tts,
        }
        output_rows.append(row)
        if idx % args.log_every == 0:
            print(json.dumps(row, ensure_ascii=False), flush=True)

    write_jsonl(args.output, output_rows)
    summary = summarize(output_rows)
    summary_path = Path(args.summary_output or str(Path(args.output).with_suffix(".summary.json")))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
