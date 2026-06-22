#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl
from s2s_omni.llm_client import ChatClient
from s2s_omni.prompts import SYSTEM_COMPRESSION, build_compression_user_prompt
from s2s_omni.schema import S2SSample
from s2s_omni.style import style_violations

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[\u4e00-\u9fff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate compressed teacher labels from teacher-request JSONL."
    )
    parser.add_argument("--input", required=True, help="Teacher requests JSONL.")
    parser.add_argument("--output", required=True, help="Teacher labels JSONL.")
    parser.add_argument("--sft-output", help="Accepted compressed SFT JSONL.")
    parser.add_argument(
        "--backend",
        default="transformers",
        choices=["transformers", "openai"],
        help="Use local Qwen3-Omni thinker or an OpenAI-compatible chat server.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-records", type=int, default=0, help="0 means all records.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-rejected", action="store_true")
    parser.add_argument("--max-over-budget-ratio", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def count_units(text: str, lang: str) -> int:
    if lang.startswith("zh"):
        return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return len(WORD_RE.findall(text or ""))


def clean_completion(text: str) -> str:
    text = (text or "").strip()
    for token in ["<|im_end|>", "<|endoftext|>"]:
        text = text.replace(token, "")
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
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        text = text[1:-1].strip()
    return " ".join(text.split())


def iter_requests(path: str | Path, max_records: int, num_shards: int, shard_index: int):
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    for idx, record in enumerate(read_jsonl(path)):
        if max_records and idx >= max_records:
            break
        if idx % num_shards != shard_index:
            continue
        yield idx, record


def existing_ids(path: str | Path, resume: bool) -> set[str]:
    path = Path(path)
    if not resume or not path.exists():
        return set()
    ids: set[str] = set()
    for record in read_jsonl(path):
        if record.get("id"):
            ids.add(str(record["id"]))
    return ids


def validate_label(sample: S2SSample, text: str, max_over_budget_ratio: float) -> dict[str, Any]:
    output_units = count_units(text, sample.tgt_lang)
    reference_units = count_units(sample.reference_translation or "", sample.tgt_lang)
    target = sample.target
    budget_units = target.max_target_chars
    if budget_units is None:
        budget_units = target.max_target_words
    reasons: list[str] = []
    if not text:
        reasons.append("empty")
    if any(marker in text for marker in ["Source language:", "Reference translation:", "Compression constraints"]):
        reasons.append("prompt_leak")
    if budget_units is not None and output_units > budget_units * max_over_budget_ratio:
        reasons.append("over_budget")
    if reference_units and output_units >= reference_units and target.mode != "faithful":
        reasons.append("not_compressed")
    reasons.extend(style_violations(text, sample.tgt_lang))
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "output_units": output_units,
        "reference_units": reference_units,
        "budget_units": budget_units,
        "output_over_budget_ratio": (
            round(output_units / budget_units, 4) if budget_units else None
        ),
        "output_reference_ratio": (
            round(output_units / reference_units, 4) if reference_units else None
        ),
    }


def build_sft_record(sample: S2SSample, answer: str) -> dict[str, Any]:
    sample.compressed_translation = answer
    return {
        "id": sample.id,
        "messages": [
            {"role": "system", "content": SYSTEM_COMPRESSION.strip()},
            {"role": "user", "content": build_compression_user_prompt(sample, include_reference=False)},
            {"role": "assistant", "content": answer},
        ],
    }


def build_teacher_messages(sample: S2SSample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_COMPRESSION.strip()},
        {"role": "user", "content": build_compression_user_prompt(sample, include_reference=True)},
    ]


class TransformersTeacher:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoConfig, Qwen3OmniMoeProcessor
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration

        self.torch = torch
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            args.model,
            trust_remote_code=True,
        )
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
        )
        self.model.eval()
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p

    def generate(self, messages: list[dict[str, str]]) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        do_sample = self.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


class OpenAITeacher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.client = ChatClient(
            base_url=args.base_url.rstrip("/"),
            api_key=args.api_key,
            model=args.model,
            timeout_s=300,
        )
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature

    def generate(self, messages: list[dict[str, str]]) -> str:
        return self.client.chat(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )


def make_teacher(args: argparse.Namespace):
    if args.backend == "transformers":
        return TransformersTeacher(args)
    return OpenAITeacher(args)


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]], mode: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")
            f.flush()


def main() -> None:
    args = parse_args()
    done = existing_ids(args.output, args.resume)
    output_mode = "a" if args.resume and Path(args.output).exists() else "w"
    sft_mode = "a" if args.resume and args.sft_output and Path(args.sft_output).exists() else "w"

    teacher = make_teacher(args)
    accepted = 0
    rejected = 0
    skipped = 0
    generated = 0

    output_path = Path(args.output)
    sft_path = Path(args.sft_output) if args.sft_output else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if sft_path:
        sft_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(output_mode, encoding="utf-8") as out_f:
        sft_f = sft_path.open(sft_mode, encoding="utf-8") if sft_path else None
        try:
            for _, record in iter_requests(
                args.input, args.max_records, args.num_shards, args.shard_index
            ):
                sample_id = str(record.get("id") or "")
                if not sample_id:
                    continue
                if sample_id in done:
                    skipped += 1
                    continue
                sample = S2SSample.from_dict(record["sample"])
                raw = teacher.generate(build_teacher_messages(sample))
                label = clean_completion(raw)
                validation = validate_label(sample, label, args.max_over_budget_ratio)
                label_record = {
                    "id": sample_id,
                    "compressed_translation": label,
                    "raw_completion": raw,
                    "validation": validation,
                    "sample": {**sample.to_dict(), "compressed_translation": label},
                }
                out_f.write(json.dumps(label_record, ensure_ascii=False) + "\n")
                out_f.flush()
                generated += 1
                if validation["accepted"]:
                    accepted += 1
                else:
                    rejected += 1
                if sft_f and (validation["accepted"] or args.keep_rejected):
                    sft_record = build_sft_record(sample, label)
                    sft_f.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
                    sft_f.flush()
                if args.log_every and generated % args.log_every == 0:
                    print(
                        json.dumps(
                            {
                                "generated": generated,
                                "accepted": accepted,
                                "rejected": rejected,
                                "skipped": skipped,
                                "last_id": sample_id,
                                "last_validation": validation,
                                "last_label_preview": label[:80],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        finally:
            if sft_f:
                sft_f.close()
    print(
        json.dumps(
            {
                "generated": generated,
                "accepted": accepted,
                "rejected": rejected,
                "skipped": skipped,
                "output": str(output_path),
                "sft_output": str(sft_path) if sft_path else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
