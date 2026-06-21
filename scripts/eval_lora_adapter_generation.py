#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl, write_jsonl
from s2s_omni.metrics import heuristic_score_sample, summarize_metric_rows
from s2s_omni.prompts import SYSTEM_COMPRESSION, build_compression_user_prompt
from s2s_omni.schema import S2SSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base vs LoRA adapter generations.")
    parser.add_argument("--input", required=True, help="Teacher request or sample JSONL.")
    parser.add_argument("--adapter", required=True, help="PEFT adapter directory.")
    parser.add_argument("--output", required=True, help="Per-sample comparison JSONL.")
    parser.add_argument("--summary", required=True, help="Summary JSON.")
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--teacher-labels", help="Optional teacher labels JSONL for side-by-side output.")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--skip-base", action="store_true", help="Only generate with adapter.")
    parser.add_argument(
        "--adapter-load-mode",
        choices=["auto", "peft", "manual-merge"],
        default="auto",
        help="Use PEFT when possible, or merge LoRA weights directly for quick eval.",
    )
    return parser.parse_args()


def load_samples(path: str, num_samples: int, seed: int) -> list[S2SSample]:
    samples: list[S2SSample] = []
    for record in read_jsonl(path):
        payload = record.get("sample") if isinstance(record.get("sample"), dict) else record
        samples.append(S2SSample.from_dict(payload))
    if num_samples > 0 and len(samples) > num_samples:
        rng = random.Random(seed)
        samples = rng.sample(samples, num_samples)
    return samples


def load_teacher_labels(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    out: dict[str, str] = {}
    for record in read_jsonl(path):
        sample_id = str(record.get("id") or "")
        text = record.get("compressed_translation")
        if sample_id and text:
            out[sample_id] = str(text)
    return out


def messages_for_sample(sample: S2SSample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_COMPRESSION.strip()},
        {
            "role": "user",
            "content": build_compression_user_prompt(sample, include_reference=False),
        },
    ]


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


def prefixed_metrics(prefix: str, row: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in row.items() if key != "id"}


class GenerationRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        model: Any,
        tokenizer: Any,
        processor: Any,
        peft_model: bool,
    ) -> None:
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.peft_model = peft_model
        self.adapter_merged = peft_model

    def ensure_adapter(self) -> None:
        if self.adapter_merged:
            return
        stats = merge_lora_adapter(self.model, self.args.adapter)
        print(json.dumps({"manual_lora_merge": stats}, ensure_ascii=False), flush=True)
        self.adapter_merged = True

    def generate(self, sample: S2SSample, use_adapter: bool) -> str:
        import torch

        if use_adapter and not self.peft_model:
            self.ensure_adapter()
        if not use_adapter and not self.peft_model and self.adapter_merged:
            raise RuntimeError("manual-merge adapter cannot be disabled after merge")

        prompt = self.processor.apply_chat_template(
            messages_for_sample(sample),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        with torch.inference_mode():
            if self.peft_model and not use_adapter:
                with self.model.disable_adapter():
                    outputs = self.model.generate(**inputs, **kwargs)
            else:
                outputs = self.model.generate(**inputs, **kwargs)
        new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
        return clean_completion(
            self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        )


def _adapter_model_path(adapter_dir: str) -> Path:
    path = Path(adapter_dir) / "adapter_model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"adapter weights not found: {path}")
    return path


def _module_name_from_lora_key(key: str) -> str:
    name = key.removeprefix("base_model.model.")
    return name.removesuffix(".lora_A.weight")


def merge_lora_adapter(model: Any, adapter_dir: str) -> dict[str, Any]:
    import torch
    from safetensors.torch import safe_open

    adapter_path = _adapter_model_path(adapter_dir)
    adapter_config = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    rank = int(adapter_config.get("r", 16))
    alpha = int(adapter_config.get("lora_alpha", 32))
    scaling = alpha / rank

    modules = dict(model.named_modules())
    merged = 0
    skipped: list[str] = []
    with safe_open(adapter_path, framework="pt", device="cpu") as tensors:
        keys = list(tensors.keys())
        for key in keys:
            if not key.endswith(".lora_A.weight"):
                continue
            module_name = _module_name_from_lora_key(key)
            module = modules.get(module_name)
            b_key = key.replace(".lora_A.weight", ".lora_B.weight")
            if module is None or b_key not in keys or not hasattr(module, "weight"):
                skipped.append(module_name)
                continue

            weight = module.weight
            device = weight.device
            a = tensors.get_tensor(key).to(device=device, dtype=torch.float32)
            b = tensors.get_tensor(b_key).to(device=device, dtype=torch.float32)
            delta = torch.matmul(b, a) * scaling
            if tuple(delta.shape) != tuple(weight.shape):
                skipped.append(module_name)
                continue
            weight.data.add_(delta.to(dtype=weight.dtype))
            merged += 1

    return {
        "merged": merged,
        "skipped": len(skipped),
        "skipped_examples": skipped[:8],
        "adapter_path": str(adapter_path),
    }


def make_generator(args: argparse.Namespace) -> GenerationRunner:
    import torch
    from transformers import AutoConfig, Qwen3OmniMoeProcessor
    from transformers import Qwen3OmniMoeThinkerForConditionalGeneration

    processor = Qwen3OmniMoeProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    full_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    base = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        args.model,
        config=full_config.thinker_config,
        trust_remote_code=True,
        dtype="auto",
        device_map=args.device_map,
    )

    if args.adapter_load_mode in {"auto", "peft"}:
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(base, args.adapter)
            model.eval()
            return GenerationRunner(args, model, tokenizer, processor, peft_model=True)
        except Exception as exc:
            if args.adapter_load_mode == "peft":
                raise
            print(
                json.dumps(
                    {
                        "adapter_load_mode": "manual-merge",
                        "peft_error": f"{type(exc).__name__}: {exc}",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    base.eval()
    return GenerationRunner(args, base, tokenizer, processor, peft_model=False)


def main() -> None:
    args = parse_args()
    samples = load_samples(args.input, args.num_samples, args.seed)
    teacher_labels = load_teacher_labels(args.teacher_labels)
    runner = make_generator(args)

    base_texts: dict[str, str] = {}
    if not args.skip_base and not runner.peft_model:
        for index, sample in enumerate(samples, start=1):
            base_texts[sample.id] = runner.generate(sample, use_adapter=False)
            print(
                json.dumps(
                    {
                        "index": index,
                        "id": sample.id,
                        "phase": "base",
                        "base_preview": base_texts[sample.id][:80],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        runner.ensure_adapter()

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        if args.skip_base:
            base_text = ""
        elif runner.peft_model:
            base_text = runner.generate(sample, use_adapter=False)
        else:
            base_text = base_texts[sample.id]
        evo_text = runner.generate(sample, use_adapter=True)
        teacher = teacher_labels.get(sample.id)
        row: dict[str, Any] = {
            "id": sample.id,
            "source_text": sample.source_text,
            "reference_translation": sample.reference_translation,
            "teacher_compressed_translation": teacher,
            "base_prediction": base_text,
            "evo_prediction": evo_text,
        }
        if base_text:
            row.update(prefixed_metrics("base", heuristic_score_sample(sample, base_text)))
        row.update(prefixed_metrics("evo", heuristic_score_sample(sample, evo_text)))
        rows.append(row)
        print(
            json.dumps(
                {
                    "index": index,
                    "id": sample.id,
                    "base_len": row.get("base_candidate_units"),
                    "base_budget": row.get("base_target_budget_ratio"),
                    "evo_len": row.get("evo_candidate_units"),
                    "evo_budget": row.get("evo_target_budget_ratio"),
                    "evo_preview": evo_text[:80],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    write_jsonl(args.output, rows)
    base_rows = [
        {"id": row["id"], **{key[len("base_") :]: value for key, value in row.items() if key.startswith("base_")}}
        for row in rows
        if row.get("base_prediction")
    ]
    evo_rows = [
        {"id": row["id"], **{key[len("evo_") :]: value for key, value in row.items() if key.startswith("evo_")}}
        for row in rows
    ]
    summary = {
        "samples": len(rows),
        "adapter": args.adapter,
        "base": summarize_metric_rows(base_rows) if base_rows else None,
        "evo": summarize_metric_rows(evo_rows),
        "output": args.output,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
