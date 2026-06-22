#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl, write_jsonl
from s2s_omni.prompts import SYSTEM_COMPRESSION, build_compression_user_prompt
from s2s_omni.schema import S2SSample
from s2s_omni.style import style_violations
from s2s_omni.tts import tts_metadata_for_backend

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[\u4e00-\u9fff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble final pass-through/compression policy SFT data."
    )
    parser.add_argument("--policy-manifest", required=True)
    parser.add_argument("--compression-labels", nargs="+", required=True)
    parser.add_argument("--output-sft", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--tts-requests-output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--reject-output")
    parser.add_argument("--target-size", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-over-budget-ratio", type=float, default=1.0)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def count_units(text: str, lang: str) -> int:
    if lang.startswith("zh"):
        return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return len(WORD_RE.findall(text or ""))


def budget_units(sample: S2SSample) -> int | None:
    return sample.target.max_target_chars or sample.target.max_target_words


def sft_record(sample: S2SSample, answer: str) -> dict[str, Any]:
    sample.compressed_translation = answer
    return {
        "id": sample.id,
        "messages": [
            {"role": "system", "content": SYSTEM_COMPRESSION.strip()},
            {"role": "user", "content": build_compression_user_prompt(sample, include_reference=False)},
            {"role": "assistant", "content": answer},
        ],
    }


def load_accepted_labels(paths: list[str] | list[Path]) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for path in paths:
        for record in read_jsonl(path):
            sample_id = str(record.get("id") or "")
            if not sample_id:
                continue
            validation = record.get("validation") or {}
            if not validation.get("accepted"):
                continue
            labels[sample_id] = record
    return labels


def final_reject_reasons(
    sample: S2SSample,
    answer: str,
    max_over_budget_ratio: float,
) -> list[str]:
    reasons: list[str] = []
    if not answer.strip():
        reasons.append("empty")
    reasons.extend(style_violations(answer, sample.tgt_lang))
    allowed = budget_units(sample)
    output_units = count_units(answer, sample.tgt_lang)
    if allowed is not None and output_units > allowed * max_over_budget_ratio:
        reasons.append("over_budget")
    return reasons


def tts_request(sample: S2SSample, answer: str) -> dict[str, Any]:
    metadata = sample.metadata
    backend = metadata.get("tts_backend", "qwen3_tts")
    default_tts_metadata = tts_metadata_for_backend(
        str(backend),
        metadata.get("tts_model_path"),
    )
    tts_metadata = {
        **default_tts_metadata,
        **{key: value for key, value in metadata.items() if key.startswith("tts_")},
    }
    unit_rate = metadata.get("default_target_unit_rate")
    output_units = count_units(answer, sample.tgt_lang)
    estimated_duration_s = None
    if unit_rate:
        estimated_duration_s = round(output_units / float(unit_rate), 3)
    return {
        "id": sample.id,
        "backend": tts_metadata.get("tts_backend", "qwen3_tts"),
        "config_cls": tts_metadata.get("tts_config_cls"),
        "model_path": tts_metadata.get("tts_model_path"),
        "relay_backend": tts_metadata.get("tts_relay_backend", "shm"),
        "sglang_omni_example": tts_metadata.get("tts_sglang_omni_example"),
        "source_audio": sample.audio_path,
        "source_text": sample.source_text,
        "target_text": answer,
        "src_lang": sample.src_lang,
        "tgt_lang": sample.tgt_lang,
        "source_speed_factor": metadata.get("source_speed_factor"),
        "source_wall_duration_s": metadata.get("source_wall_duration_s"),
        "target_duration_budget_s": sample.target.max_target_duration_s,
        "estimated_target_speech_s": estimated_duration_s,
        "target_text_units": output_units,
        "target_unit_rate": unit_rate,
        "duration_policy": tts_metadata.get(
            "tts_duration_policy",
            "match_target_text_default_speech_duration",
        ),
    }


def manifest_record(sample: S2SSample, answer: str, source: str) -> dict[str, Any]:
    sample.compressed_translation = answer
    out = sample.to_dict()
    metadata = dict(out.get("metadata") or {})
    metadata["final_sft_source"] = source
    metadata["final_target_units"] = count_units(answer, sample.tgt_lang)
    out["metadata"] = metadata
    return out


def candidate_from_policy_record(
    record: dict[str, Any],
    labels: dict[str, dict[str, Any]],
) -> tuple[S2SSample | None, str, str, str | None]:
    sample = S2SSample.from_dict(record)
    kind = str(sample.metadata.get("policy_sample_kind") or "")
    if kind == "pass_through":
        answer = sample.reference_translation or sample.compressed_translation or ""
        return sample, answer, "pass_through", None
    if kind == "compression":
        label = labels.get(sample.id)
        if label is None:
            return sample, "", "compression", "missing_or_rejected_teacher_label"
        answer = str(label.get("compressed_translation") or "").strip()
        labeled_sample = S2SSample.from_dict(label["sample"])
        return labeled_sample, answer, "compression", None
    return sample, "", kind or "unknown", "unknown_policy_sample_kind"


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    labels = load_accepted_labels(args.compression_labels)
    policy_records = read_jsonl(args.policy_manifest)
    rng.shuffle(policy_records)

    sft_records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    tts_requests: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    reject_counts: Counter[str] = Counter()

    for record in policy_records:
        if len(sft_records) >= args.target_size:
            break
        sample, answer, source, early_reject = candidate_from_policy_record(record, labels)
        if sample is None:
            continue
        reasons = [early_reject] if early_reject else []
        if not reasons:
            reasons = final_reject_reasons(sample, answer, args.max_over_budget_ratio)
        if reasons:
            rejected.append({"id": sample.id, "source": source, "reasons": reasons})
            for reason in reasons:
                reject_counts[str(reason).split(":", 1)[0]] += 1
            continue
        sft_records.append(sft_record(sample, answer))
        manifests.append(manifest_record(sample, answer, source))
        tts_requests.append(tts_request(sample, answer))
        source_counts[source] += 1

    if len(sft_records) < args.target_size and not args.allow_incomplete:
        raise RuntimeError(
            f"assembled {len(sft_records)} records, below target {args.target_size}"
        )

    write_jsonl(args.output_sft, sft_records)
    write_jsonl(args.output_manifest, manifests)
    write_jsonl(args.tts_requests_output, tts_requests)
    if args.reject_output:
        write_jsonl(args.reject_output, rejected)

    summary = {
        "policy_manifest": args.policy_manifest,
        "compression_labels": args.compression_labels,
        "target_size": args.target_size,
        "assembled": len(sft_records),
        "source_counts": dict(source_counts),
        "rejected": len(rejected),
        "reject_counts": dict(reject_counts),
        "output_sft": args.output_sft,
        "output_manifest": args.output_manifest,
        "tts_requests_output": args.tts_requests_output,
    }
    summary_path = args.summary_output
    if summary_path is None:
        summary_path = str(Path(args.output_sft).with_suffix(".summary.json"))
    Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
