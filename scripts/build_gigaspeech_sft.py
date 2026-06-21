#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import write_jsonl
from s2s_omni.prompts import SYSTEM_COMPRESSION, build_compression_user_prompt
from s2s_omni.schema import CompressionTarget, S2SSample, Timing

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[\u4e00-\u9fff]")
GIGASPEECH_TSV = (
    "/mnt/taurus/data/siqiouyang/datasets/gigaspeech/"
    "train_xl_case_ft-qwen2.5-32b-instruct_marked_mfa_punc_asr.tsv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Qwen3-Omni text-side SFT data from GigaSpeech S2TT TSV. "
            "The script emits faithful warm-up records plus stress cases that need "
            "teacher compression labels."
        )
    )
    parser.add_argument("--tsv", default=GIGASPEECH_TSV)
    parser.add_argument("--out-dir", default="work/gigaspeech")
    parser.add_argument("--max-rows", type=int, default=0, help="0 means scan all rows.")
    parser.add_argument("--faithful-limit", type=int, default=20000)
    parser.add_argument("--teacher-limit", type=int, default=5000)
    parser.add_argument(
        "--split-ratios",
        default="train=0.9,dev=0.05,test=0.05",
        help=(
            "Base-id split ratios. Accepts 'train=0.9,dev=0.05,test=0.05' "
            "or '0.9,0.05,0.05' for train/dev/test."
        ),
    )
    parser.add_argument(
        "--dev-teacher-limit",
        type=int,
        default=300,
        help="Held-out dev teacher-request limit. 0 disables dev teacher output.",
    )
    parser.add_argument(
        "--test-teacher-limit",
        type=int,
        default=300,
        help="Held-out test teacher-request limit. 0 disables test teacher output.",
    )
    parser.add_argument(
        "--dev-faithful-limit",
        type=int,
        default=0,
        help="Optional held-out faithful examples. Usually not needed for SFT.",
    )
    parser.add_argument(
        "--test-faithful-limit",
        type=int,
        default=0,
        help="Optional held-out faithful examples. Usually not needed for SFT.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sample-rate", type=float, default=16000.0)
    parser.add_argument("--speed-factors", default="1.0,1.35,1.7,2.0")
    parser.add_argument("--max-end-lag-s", type=float, default=1.5)
    parser.add_argument(
        "--zh-char-per-sec",
        type=float,
        default=5.0,
        help="Approximate comfortable Mandarin TTS budget for target characters.",
    )
    parser.add_argument(
        "--word-per-sec",
        type=float,
        default=2.75,
        help="Fallback target-language word budget for non-CJK outputs.",
    )
    parser.add_argument("--stress-ratio-threshold", type=float, default=0.82)
    parser.add_argument("--min-src-units", type=int, default=8)
    parser.add_argument("--min-tgt-units", type=int, default=8)
    parser.add_argument(
        "--faithful-keep-prob",
        type=float,
        default=0.2,
        help="Reservoir candidate probability for faithful warm-up examples.",
    )
    return parser.parse_args()


def parse_split_ratios(raw: str) -> list[tuple[str, float]]:
    raw = raw.strip()
    if not raw:
        raise ValueError("--split-ratios cannot be empty")

    if "=" in raw:
        pairs: list[tuple[str, float]] = []
        for item in raw.split(","):
            name, value = item.split("=", 1)
            name = name.strip()
            if not name:
                raise ValueError(f"empty split name in --split-ratios: {raw}")
            pairs.append((name, float(value)))
    else:
        names = ["train", "dev", "test"]
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
        if len(values) != len(names):
            raise ValueError("unnamed --split-ratios must have train,dev,test values")
        pairs = list(zip(names, values, strict=True))

    total = sum(value for _, value in pairs)
    if total <= 0:
        raise ValueError("--split-ratios must sum to a positive value")
    return [(name, max(0.0, value) / total) for name, value in pairs]


def assign_split(base_id: str, split_ratios: list[tuple[str, float]], seed: int) -> str:
    key = f"{seed}:{base_id}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    score = int.from_bytes(digest, "big") / float(1 << 64)
    cumulative = 0.0
    fallback = split_ratios[-1][0]
    for name, ratio in split_ratios:
        if ratio <= 0:
            continue
        fallback = name
        cumulative += ratio
        if score < cumulative:
            return name
    return fallback


def count_units(text: str, lang: str) -> int:
    if lang.startswith("zh"):
        return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return len(WORD_RE.findall(text or ""))


def clean_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def target_unit_rate(lang: str, zh_char_per_sec: float, word_per_sec: float) -> float:
    return zh_char_per_sec if lang.startswith("zh") else word_per_sec


def parse_trajectory(raw: str) -> list[str]:
    try:
        value = ast.literal_eval(raw)
    except Exception:
        return []
    if not isinstance(value, list):
        return []
    return [clean_text(str(item)) for item in value if clean_text(str(item))]


def make_sample_from_row(
    row: dict[str, str],
    mode: str,
    target_ratio: float,
    max_target_units: int | None,
    speed_factor: float,
    duration_s: float,
    max_end_lag_s: float,
    max_target_wpm: float | None = None,
) -> S2SSample:
    src = clean_text(row["src_text"])
    tgt = clean_text(row["tgt_text"])
    tgt_lang = row.get("tgt_lang", "zh")
    scaled_duration = duration_s / speed_factor
    sample_id = row["id"] if mode == "faithful" else f"{row['id']}__speed_{speed_factor:g}"
    trajectory = parse_trajectory(row.get("trajectory", ""))
    max_target_chars = max_target_units if tgt_lang.startswith("zh") else None
    max_target_words = None if tgt_lang.startswith("zh") else max_target_units
    target = CompressionTarget(
        mode=mode,
        target_ratio=target_ratio,
        max_target_chars=max_target_chars,
        max_target_words=max_target_words,
        max_target_duration_s=round(scaled_duration + max_end_lag_s, 3),
        max_target_wpm=max_target_wpm,
        max_end_lag_s=max_end_lag_s,
        preserve=[
            "named_entities",
            "numbers",
            "terminology",
            "negation",
            "modality",
            "causality",
            "contrast",
        ],
        compressible=[
            "discourse_fillers",
            "repeated_explanations",
            "redundant_modifiers",
            "low_information_examples",
            "self_repairs",
        ],
    )
    return S2SSample(
        id=sample_id,
        src_lang=row.get("src_lang", "en"),
        tgt_lang=tgt_lang,
        source_text=src,
        reference_translation=tgt,
        compressed_translation=tgt if mode == "faithful" else None,
        audio_path=row.get("audio") or None,
        target=target,
        timing=Timing(src_start_s=0.0, src_end_s=round(scaled_duration, 3)),
        metadata={
            "base_id": row["id"],
            "n_frames": int(row["n_frames"]) if row.get("n_frames") else None,
            "source_speed_factor": speed_factor,
            "target_units": count_units(tgt, row.get("tgt_lang", "zh")),
            "trajectory_nonempty": len(trajectory),
            "trajectory_preview": trajectory[:8],
        },
    )


def sft_record(sample: S2SSample) -> dict[str, Any]:
    return {
        "id": sample.id,
        "messages": [
            {"role": "system", "content": SYSTEM_COMPRESSION.strip()},
            {"role": "user", "content": build_compression_user_prompt(sample, include_reference=False)},
            {"role": "assistant", "content": sample.compressed_translation or ""},
        ],
    }


def teacher_record(sample: S2SSample) -> dict[str, Any]:
    return {
        "id": sample.id,
        "messages": [
            {"role": "system", "content": SYSTEM_COMPRESSION.strip()},
            {"role": "user", "content": build_compression_user_prompt(sample, include_reference=True)},
        ],
        "sample": sample.to_dict(),
    }


def reservoir_add(items: list[Any], item: Any, limit: int, seen: int, rng: random.Random) -> None:
    if limit <= 0:
        return
    if len(items) < limit:
        items.append(item)
        return
    j = rng.randint(0, seen - 1)
    if j < limit:
        items[j] = item


def empty_split_bucket() -> dict[str, Any]:
    return {
        "faithful": [],
        "teacher": [],
        "seen_faithful": 0,
        "seen_teacher": 0,
        "usable_rows": 0,
        "base_ids": set(),
    }


def write_split_outputs(
    out_dir: Path,
    split_name: str,
    faithful: list[S2SSample],
    teacher: list[S2SSample],
) -> dict[str, str]:
    split_dir = out_dir / "splits" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    faithful_manifest = split_dir / "faithful_manifest.jsonl"
    faithful_sft = split_dir / "faithful_sft.jsonl"
    teacher_manifest = split_dir / "compression_teacher_manifest.jsonl"
    teacher_requests = split_dir / "compression_teacher_requests.jsonl"

    write_jsonl(faithful_manifest, (sample.to_dict() for sample in faithful))
    write_jsonl(faithful_sft, (sft_record(sample) for sample in faithful))
    write_jsonl(teacher_manifest, (sample.to_dict() for sample in teacher))
    write_jsonl(teacher_requests, (teacher_record(sample) for sample in teacher))
    return {
        "faithful_manifest": str(faithful_manifest),
        "faithful_sft": str(faithful_sft),
        "teacher_manifest": str(teacher_manifest),
        "teacher_requests": str(teacher_requests),
    }


def write_train_alias_outputs(
    out_dir: Path,
    faithful: list[S2SSample],
    teacher: list[S2SSample],
) -> dict[str, str]:
    faithful_manifest = out_dir / "faithful_manifest.jsonl"
    faithful_sft = out_dir / "faithful_sft.jsonl"
    teacher_manifest = out_dir / "compression_teacher_manifest.jsonl"
    teacher_requests = out_dir / "compression_teacher_requests.jsonl"

    write_jsonl(faithful_manifest, (sample.to_dict() for sample in faithful))
    write_jsonl(faithful_sft, (sft_record(sample) for sample in faithful))
    write_jsonl(teacher_manifest, (sample.to_dict() for sample in teacher))
    write_jsonl(teacher_requests, (teacher_record(sample) for sample in teacher))
    return {
        "faithful_manifest": str(faithful_manifest),
        "faithful_sft": str(faithful_sft),
        "teacher_manifest": str(teacher_manifest),
        "teacher_requests": str(teacher_requests),
    }


def pairwise_base_id_overlaps(buckets: dict[str, dict[str, Any]]) -> dict[str, int]:
    names = list(buckets)
    overlaps: dict[str, int] = {}
    for i, left in enumerate(names):
        left_ids = buckets[left]["base_ids"]
        for right in names[i + 1 :]:
            overlaps[f"{left}__{right}"] = len(left_ids & buckets[right]["base_ids"])
    return overlaps


def iter_rows(path: str, max_rows: int) -> Iterable[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader, start=1):
            if max_rows and idx > max_rows:
                break
            yield row


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    speed_factors = [float(x.strip()) for x in args.speed_factors.split(",") if x.strip()]
    split_ratios = parse_split_ratios(args.split_ratios)
    split_names = [name for name, ratio in split_ratios if ratio > 0]
    buckets = {name: empty_split_bucket() for name in split_names}
    teacher_limits = {
        "train": args.teacher_limit,
        "dev": args.dev_teacher_limit,
        "test": args.test_teacher_limit,
    }
    faithful_limits = {
        "train": args.faithful_limit,
        "dev": args.dev_faithful_limit,
        "test": args.test_faithful_limit,
    }
    scanned = 0
    usable = 0

    for row in iter_rows(args.tsv, args.max_rows):
        scanned += 1
        src = clean_text(row.get("src_text", ""))
        tgt = clean_text(row.get("tgt_text", ""))
        if not src or not tgt:
            continue
        src_lang = row.get("src_lang", "en")
        tgt_lang = row.get("tgt_lang", "zh")
        src_units = count_units(src, src_lang)
        tgt_units = count_units(tgt, tgt_lang)
        if src_units < args.min_src_units or tgt_units < args.min_tgt_units:
            continue
        try:
            duration_s = float(row["n_frames"]) / args.sample_rate
        except Exception:
            continue
        if duration_s <= 0:
            continue
        usable += 1
        base_id = row["id"]
        split_name = assign_split(base_id, split_ratios, args.seed)
        bucket = buckets[split_name]
        bucket["usable_rows"] += 1
        bucket["base_ids"].add(base_id)

        if rng.random() < args.faithful_keep_prob:
            bucket["seen_faithful"] += 1
            sample = make_sample_from_row(
                row,
                mode="faithful",
                target_ratio=1.0,
                max_target_units=None,
                speed_factor=1.0,
                duration_s=duration_s,
                max_end_lag_s=args.max_end_lag_s,
            )
            reservoir_add(
                bucket["faithful"],
                sample,
                faithful_limits.get(split_name, 0),
                bucket["seen_faithful"],
                rng,
            )

        rate = target_unit_rate(tgt_lang, args.zh_char_per_sec, args.word_per_sec)
        for speed in speed_factors:
            scaled_duration = duration_s / speed
            budget_units = int((scaled_duration + args.max_end_lag_s) * rate)
            ratio = budget_units / max(1, tgt_units)
            if ratio >= args.stress_ratio_threshold:
                continue
            bucket["seen_teacher"] += 1
            sample = make_sample_from_row(
                row,
                mode="concise" if ratio >= 0.62 else "very_concise",
                target_ratio=round(max(0.35, min(0.95, ratio)), 3),
                max_target_units=max(4, budget_units),
                speed_factor=speed,
                duration_s=duration_s,
                max_end_lag_s=args.max_end_lag_s,
            )
            reservoir_add(
                bucket["teacher"],
                sample,
                teacher_limits.get(split_name, 0),
                bucket["seen_teacher"],
                rng,
            )

    summary_path = out_dir / "build_summary.json"
    split_outputs: dict[str, dict[str, str]] = {}
    for split_name in split_names:
        split_outputs[split_name] = write_split_outputs(
            out_dir,
            split_name,
            buckets[split_name]["faithful"],
            buckets[split_name]["teacher"],
        )

    train_alias_outputs = {}
    if "train" in buckets:
        train_alias_outputs = write_train_alias_outputs(
            out_dir,
            buckets["train"]["faithful"],
            buckets["train"]["teacher"],
        )

    summary = {
        "tsv": args.tsv,
        "scanned_rows": scanned,
        "usable_rows": usable,
        "split_seed": args.seed,
        "split_ratios": dict(split_ratios),
        "split_unit": "base_id",
        "split_limits": {
            "teacher": {name: teacher_limits.get(name, 0) for name in split_names},
            "faithful": {name: faithful_limits.get(name, 0) for name in split_names},
        },
        "split_stats": {
            name: {
                "base_ids": len(bucket["base_ids"]),
                "usable_rows": bucket["usable_rows"],
                "faithful_records": len(bucket["faithful"]),
                "faithful_seen_candidates": bucket["seen_faithful"],
                "teacher_records": len(bucket["teacher"]),
                "teacher_seen_candidates": bucket["seen_teacher"],
            }
            for name, bucket in buckets.items()
        },
        "base_id_overlap_check": pairwise_base_id_overlaps(buckets),
        "speed_factors": speed_factors,
        "stress_ratio_threshold": args.stress_ratio_threshold,
        "outputs": split_outputs,
        "train_alias_outputs": train_alias_outputs,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
