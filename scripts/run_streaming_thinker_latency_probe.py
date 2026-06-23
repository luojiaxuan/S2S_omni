#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
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
from s2s_omni.schema import CompressionTarget, S2SSample, Timing


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\s]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe thinker-only LoRA latency on simulated streaming source chunks."
    )
    parser.add_argument("--input", required=True, help="Prediction/sample JSONL.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=260622)
    parser.add_argument("--chunk-s", type=float, default=2.0)
    parser.add_argument("--max-ticks-per-sample", type=int, default=8)
    parser.add_argument("--context-mode", choices=["chunk", "prefix"], default="chunk")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--default-target-unit-rate", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def load_records(path: str, num_samples: int, seed: int) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    if num_samples > 0 and len(rows) > num_samples:
        rows = random.Random(seed).sample(rows, num_samples)
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


def source_budget_s(sample: S2SSample, record: dict[str, Any]) -> float:
    for prefix in ["evo", "base"]:
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


def split_source_text(text: str, source_duration_s: float, chunk_s: float) -> list[dict[str, Any]]:
    tokens = TOKEN_RE.findall(text or "")
    if not tokens:
        return []
    tick_count = max(1, int(math.ceil(source_duration_s / chunk_s)))
    chunks: list[dict[str, Any]] = []
    for tick in range(tick_count):
        start = int(round(len(tokens) * tick / tick_count))
        end = int(round(len(tokens) * (tick + 1) / tick_count))
        if end <= start:
            end = min(len(tokens), start + 1)
        chunks.append(
            {
                "tick_index": tick,
                "start_s": round(tick * chunk_s, 6),
                "end_s": round(min(source_duration_s, (tick + 1) * chunk_s), 6),
                "chunk_text": detokenize(tokens[start:end]),
                "prefix_text": detokenize(tokens[:end]),
            }
        )
    return chunks


def detokenize(tokens: list[str]) -> str:
    out = ""
    for token in tokens:
        if not out:
            out = token
        elif re.match(r"^[A-Za-z0-9]", token) and re.search(r"[A-Za-z0-9]$", out):
            out += " " + token
        elif re.match(r"^[A-Za-z0-9]", token):
            out += " " + token
        else:
            out += token
    return out


def chunk_sample(
    sample: S2SSample,
    chunk: dict[str, Any],
    source_text: str,
    budget_s: float,
    rate: float,
) -> S2SSample:
    target_units = max(1, int(math.floor(budget_s * rate)))
    metadata = {
        **sample.metadata,
        "streaming_probe": True,
        "source_wall_duration_s": budget_s,
        "playback_budget_s": budget_s,
        "allowed_speech_s": budget_s,
        "default_target_unit_rate": rate,
        "rtf_threshold": 1.0,
        "chunk_start_s": chunk["start_s"],
        "chunk_end_s": chunk["end_s"],
    }
    return S2SSample(
        id=f"{sample.id}__tick_{chunk['tick_index']:04d}",
        src_lang=sample.src_lang,
        tgt_lang=sample.tgt_lang,
        source_text=source_text,
        reference_translation=None,
        compressed_translation=None,
        candidate_translation=None,
        audio_path=sample.audio_path,
        target=CompressionTarget(
            mode="concise",
            max_target_chars=target_units if sample.tgt_lang.startswith("zh") else None,
            max_target_words=None if sample.tgt_lang.startswith("zh") else target_units,
            max_target_duration_s=budget_s,
            max_end_lag_s=0.0,
        ),
        timing=Timing(src_start_s=chunk["start_s"], src_end_s=chunk["end_s"]),
        metadata=metadata,
    )


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


class ThinkerRunner:
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
        self.model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            args.model,
            config=full_config.thinker_config,
            trust_remote_code=True,
            dtype="auto",
            device_map=args.device_map,
        ).eval()
        self.merge_lora_adapter(args.adapter)
        self.max_new_tokens = args.max_new_tokens

    def merge_lora_adapter(self, adapter_dir: str) -> None:
        from safetensors.torch import safe_open

        adapter_path = Path(adapter_dir) / "adapter_model.safetensors"
        adapter_config = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
        rank = int(adapter_config.get("r", 16))
        alpha = int(adapter_config.get("lora_alpha", 32))
        scaling = alpha / rank
        modules = dict(self.model.named_modules())
        merged = 0
        skipped = 0
        with safe_open(adapter_path, framework="pt", device="cpu") as tensors:
            keys = list(tensors.keys())
            for key in keys:
                if not key.endswith(".lora_A.weight"):
                    continue
                module_name = key.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
                module = modules.get(module_name)
                b_key = key.replace(".lora_A.weight", ".lora_B.weight")
                if module is None or b_key not in keys or not hasattr(module, "weight"):
                    skipped += 1
                    continue
                weight = module.weight
                a = tensors.get_tensor(key).to(device=weight.device, dtype=self.torch.float32)
                b = tensors.get_tensor(b_key).to(device=weight.device, dtype=self.torch.float32)
                delta = self.torch.matmul(b, a) * scaling
                if tuple(delta.shape) != tuple(weight.shape):
                    skipped += 1
                    continue
                weight.data.add_(delta.to(dtype=weight.dtype))
                merged += 1
        print(json.dumps({"manual_lora_merge": {"merged": merged, "skipped": skipped}}, ensure_ascii=False), flush=True)

    def generate(self, sample: S2SSample) -> dict[str, Any]:
        prompt = self.processor.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_COMPRESSION.strip()},
                {"role": "user", "content": build_compression_user_prompt(sample, include_reference=False)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, **kwargs)
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        wall_s = time.perf_counter() - started
        new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
        text = clean_completion(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return {
            "text": text,
            "wall_s": round(wall_s, 6),
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "generated_tokens": int(new_tokens.shape[0]),
        }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "samples": len({row["base_id"] for row in rows}),
        "chunk_s": rows[0]["chunk_s"] if rows else None,
        "context_mode": rows[0]["context_mode"] if rows else None,
        "wall_s": stats([row["thinker_wall_s"] for row in rows]),
        "deadline_ratio": stats([row["thinker_deadline_ratio"] for row in rows]),
        "generated_tokens": stats([row["generated_tokens"] for row in rows]),
        "output_units": stats([row["output_units"] for row in rows]),
        "deadline_miss": sum(1 for row in rows if not row["thinker_fits_chunk"]),
    }


def stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    values = sorted(float(value) for value in values)
    return {
        "min": round(values[0], 6),
        "p50": round(values[len(values) // 2], 6),
        "p90": round(values[min(len(values) - 1, int(len(values) * 0.9))], 6),
        "max": round(values[-1], 6),
        "mean": round(sum(values) / len(values), 6),
    }


def main() -> None:
    args = parse_args()
    runner = ThinkerRunner(args)
    rows: list[dict[str, Any]] = []
    for record in load_records(args.input, args.num_samples, args.seed):
        sample = sample_from_record(record)
        total_budget_s = source_budget_s(sample, record)
        chunks = split_source_text(sample.source_text, total_budget_s, args.chunk_s)
        if args.max_ticks_per_sample > 0:
            chunks = chunks[: args.max_ticks_per_sample]
        for chunk in chunks:
            source_text = chunk[f"{args.context_mode}_text"]
            budget_s = max(1.0e-6, chunk["end_s"] - chunk["start_s"])
            probe_sample = chunk_sample(
                sample,
                chunk,
                source_text,
                budget_s,
                args.default_target_unit_rate,
            )
            result = runner.generate(probe_sample)
            output_units = unit_count(result["text"], probe_sample.tgt_lang)
            row = {
                "base_id": sample.id,
                "id": probe_sample.id,
                "chunk_s": args.chunk_s,
                "context_mode": args.context_mode,
                "tick_index": chunk["tick_index"],
                "chunk_start_s": chunk["start_s"],
                "chunk_end_s": chunk["end_s"],
                "chunk_budget_s": round(budget_s, 6),
                "source_context_chars": len(source_text),
                "source_context_text": source_text,
                "target_text": result["text"],
                "output_units": output_units,
                "thinker_wall_s": result["wall_s"],
                "thinker_deadline_ratio": round(result["wall_s"] / budget_s, 6),
                "thinker_fits_chunk": result["wall_s"] <= budget_s,
                "prompt_tokens": result["prompt_tokens"],
                "generated_tokens": result["generated_tokens"],
            }
            rows.append(row)
            if len(rows) % args.log_every == 0:
                print(json.dumps(row, ensure_ascii=False), flush=True)
    write_jsonl(args.output, rows)
    summary = summarize(rows)
    summary_path = Path(args.summary_output or str(Path(args.output).with_suffix(".summary.json")))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
