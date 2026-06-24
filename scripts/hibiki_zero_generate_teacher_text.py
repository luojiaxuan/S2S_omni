#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.hibiki_zero import (
    HibikiSample,
    compact_whitespace,
    english_words,
    join_chunk_texts,
)
from s2s_omni.io import read_jsonl
from s2s_omni.llm_client import ChatClient, extract_json_object


SYSTEM_PROMPT = """You are a simultaneous speech-to-speech translation data teacher.
Translate the source speech into concise, natural spoken English for streaming playback.
Shorten only when needed to fit each chunk duration budget. Preserve names, numbers, entities, negation, and decisions. Remove filler and non-critical repetition.
Return JSON only with this schema:
{"full_text":"...","chunks":[{"chunk_index":0,"text":"..."},{"chunk_index":1,"text":"..."}]}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate backlog-aware English teacher text for Hibiki-Zero S2S data."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rejected-output", default="")
    parser.add_argument(
        "--backend",
        choices=["openai", "transformers_omni", "reference"],
        default="openai",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--omni-use-audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--include-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def existing_ids(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    out = set()
    for row in read_jsonl(path):
        if row.get("sample_id"):
            out.add(str(row["sample_id"]))
    return out


def prompt_for_sample(sample: HibikiSample, include_reference: bool) -> str:
    lines = [
        f"Source language: {sample.src_lang}",
        "Target language: English",
        "Chunk budgets are seconds of target speech allowed before the next source chunk.",
        "Write one English chunk per source chunk. A chunk may be empty only when speaking would be premature.",
        "",
        "Chunks:",
    ]
    for chunk in sample.source_audio_chunks:
        budget = ""
        if chunk.source_duration_s is not None:
            budget = f", duration_budget_s={chunk.source_duration_s:.3f}"
        lines.append(f"- chunk_index={chunk.index}{budget}")
        if chunk.source_text:
            lines.append(f"  source_text: {chunk.source_text}")
        if include_reference and chunk.reference_en_text:
            lines.append(f"  reference_en_text: {chunk.reference_en_text}")
    if sample.source_text:
        lines.extend(["", f"Full source transcript: {sample.source_text}"])
    if include_reference and sample.reference_en_text:
        lines.extend(["", f"Full reference English: {sample.reference_en_text}"])
    lines.append("")
    lines.append("Return compact JSON only.")
    return "\n".join(lines)


def text_messages(sample: HibikiSample, include_reference: bool) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_for_sample(sample, include_reference)},
    ]


def omni_messages(sample: HibikiSample, include_reference: bool) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = []
    for chunk in sample.source_audio_chunks:
        content.append({"type": "audio", "audio": chunk.source_audio})
    content.append({"type": "text", "text": prompt_for_sample(sample, include_reference)})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def clean_teacher_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def parse_teacher_output(raw_text: str, chunk_count: int) -> tuple[str, list[str], dict[str, Any]]:
    raw_text = clean_teacher_text(raw_text)
    data = extract_json_object(raw_text)
    chunk_rows = data.get("chunks")
    if not isinstance(chunk_rows, list):
        raise ValueError("teacher JSON missing chunks list")
    chunk_texts = [""] * chunk_count
    for item in chunk_rows:
        if not isinstance(item, dict):
            raise ValueError("teacher chunks must be objects")
        idx = int(item.get("chunk_index", item.get("index", len(chunk_texts))))
        if idx < 0 or idx >= chunk_count:
            raise ValueError(f"teacher chunk index out of range: {idx}")
        chunk_texts[idx] = compact_whitespace(str(item.get("text") or ""))
    full_text = compact_whitespace(str(data.get("full_text") or data.get("compressed_en_text") or ""))
    joined = join_chunk_texts(chunk_texts)
    if not full_text:
        full_text = joined
    if joined and english_words(joined) != english_words(full_text):
        full_text = joined
    return full_text, chunk_texts, data


def validate_teacher(sample: HibikiSample, full_text: str, chunk_texts: list[str]) -> list[str]:
    reasons: list[str] = []
    if not english_words(full_text):
        reasons.append("empty_full_text")
    if len(chunk_texts) != sample.chunk_count:
        reasons.append("chunk_count_mismatch")
    source_durations = sample.source_durations_s()
    for idx, text in enumerate(chunk_texts):
        words = len(english_words(text))
        source_s = source_durations[idx] if idx < len(source_durations) else None
        if source_s and source_s > 0 and words:
            estimated_s = words / 2.8
            if estimated_s / source_s > 1.25:
                reasons.append(f"chunk_{idx}_estimated_over_budget")
    return reasons


class OpenAITeacher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.client = ChatClient(
            base_url=args.base_url.rstrip("/"),
            api_key=args.api_key,
            model=args.model,
            timeout_s=600,
        )
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature

    def generate(self, sample: HibikiSample, include_reference: bool) -> str:
        return self.client.chat(
            text_messages(sample, include_reference),
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            response_format={"type": "json_object"},
        )


class ReferenceTeacher:
    def generate(self, sample: HibikiSample, include_reference: bool) -> str:
        chunks = []
        for chunk in sample.source_audio_chunks:
            chunks.append({"chunk_index": chunk.index, "text": chunk.reference_en_text})
        full_text = join_chunk_texts([chunk["text"] for chunk in chunks]) or sample.reference_en_text
        return json.dumps({"full_text": full_text, "chunks": chunks}, ensure_ascii=False)


class TransformersOmniTeacher:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from qwen_omni_utils import process_mm_info
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        self.torch = torch
        self.process_mm_info = process_mm_info
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(args.model, trust_remote_code=True)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            args.model,
            trust_remote_code=True,
            dtype="auto",
            device_map=args.device_map,
        )
        self.model.eval()
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature
        self.use_audio = args.omni_use_audio

    def generate(self, sample: HibikiSample, include_reference: bool) -> str:
        messages = (
            omni_messages(sample, include_reference)
            if self.use_audio
            else text_messages(sample, include_reference)
        )
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = self.process_mm_info(messages, use_audio_in_video=False)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        device = next(self.model.parameters()).device
        dtype = getattr(self.model, "dtype", None)
        moved = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                if dtype is not None and getattr(value, "is_floating_point", lambda: False)():
                    moved[key] = value.to(device=device, dtype=dtype)
                else:
                    moved[key] = value.to(device=device)
            else:
                moved[key] = value
        do_sample = self.temperature > 0
        with self.torch.inference_mode():
            output = self.model.generate(
                **moved,
                thinker_return_dict_in_generate=True,
                thinker_max_new_tokens=self.max_new_tokens,
                thinker_do_sample=do_sample,
                return_audio=False,
                use_audio_in_video=False,
            )
        text_ids = output[0] if isinstance(output, tuple) else output
        sequences = text_ids.sequences if hasattr(text_ids, "sequences") else text_ids
        new_ids = sequences[:, moved["input_ids"].shape[1] :]
        return self.processor.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def make_teacher(args: argparse.Namespace) -> OpenAITeacher | TransformersOmniTeacher | ReferenceTeacher:
    if args.backend == "reference":
        return ReferenceTeacher()
    if args.backend == "transformers_omni":
        return TransformersOmniTeacher(args)
    return OpenAITeacher(args)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
        handle.flush()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    rejected_path = Path(args.rejected_output) if args.rejected_output else output_path.with_suffix(".rejected.jsonl")
    if not args.resume:
        for path in [output_path, rejected_path]:
            if path.exists():
                path.unlink()
    done = existing_ids(output_path) | existing_ids(rejected_path)
    teacher = make_teacher(args)
    accepted = 0
    rejected = 0
    skipped = 0

    for idx, record in enumerate(read_jsonl(args.input)):
        if args.max_records and idx >= args.max_records:
            break
        try:
            sample = HibikiSample.from_dict(record)
            if sample.sample_id in done:
                skipped += 1
                continue
            raw = teacher.generate(sample, args.include_reference)
            full_text, chunk_texts, teacher_json = parse_teacher_output(raw, sample.chunk_count)
            reasons = validate_teacher(sample, full_text, chunk_texts)
            out = sample.to_dict()
            out.update(
                {
                    "compressed_en_text": full_text,
                    "teacher_chunks": [
                        {"chunk_index": i, "text": text} for i, text in enumerate(chunk_texts)
                    ],
                    "teacher_backend": args.backend,
                    "teacher_model": args.model,
                    "teacher_raw_json": teacher_json,
                    "teacher_quality_gates": {"accepted": not reasons, "reasons": reasons},
                }
            )
            out["source_audio_chunks"] = [
                {**chunk.to_dict(), "compressed_en_text": chunk_texts[i]}
                for i, chunk in enumerate(sample.source_audio_chunks)
            ]
            if reasons:
                append_jsonl(rejected_path, out)
                rejected += 1
            else:
                append_jsonl(output_path, out)
                accepted += 1
        except Exception as exc:
            rejected += 1
            append_jsonl(
                rejected_path,
                {
                    "sample_id": record.get("sample_id") or record.get("id"),
                    "accepted": False,
                    "reject_reasons": [f"exception:{type(exc).__name__}"],
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        if args.log_every > 0 and (accepted + rejected + skipped) % args.log_every == 0:
            print(
                json.dumps(
                    {"accepted": accepted, "rejected": rejected, "skipped": skipped},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    summary = {
        "input": args.input,
        "output": str(output_path),
        "rejected_output": str(rejected_path),
        "accepted": accepted,
        "rejected": rejected,
        "skipped": skipped,
        "backend": args.backend,
        "model": args.model,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
