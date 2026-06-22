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
from s2s_omni.tts import tts_metadata_for_backend

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[\u4e00-\u9fff]")
GIGASPEECH_TSV = (
    "/mnt/taurus/data/siqiouyang/datasets/gigaspeech/"
    "train_xl_case_ft-qwen2.5-32b-instruct_marked_mfa_punc_asr.tsv"
)
RTF_BIN_EDGES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]


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
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Scan all candidate decisions and write build_summary.json only.",
    )
    parser.add_argument(
        "--policy-sample-size",
        type=int,
        default=0,
        help=(
            "Train split total reservoir sample over all RTF decisions. This keeps "
            "the natural pass-through/compression mix instead of class-specific caps."
        ),
    )
    parser.add_argument("--dev-policy-sample-size", type=int, default=0)
    parser.add_argument("--test-policy-sample-size", type=int, default=0)
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
        "--rtf-threshold",
        type=float,
        default=1.0,
        help=(
            "Compress only when default-speed target speech would exceed this "
            "source-chunk real-time factor."
        ),
    )
    parser.add_argument(
        "--rtf-lag-slack-s",
        type=float,
        default=0.0,
        help=(
            "Optional slack added to the source chunk duration for RTF budget "
            "calculation. Default 0 means target speech must finish before the "
            "next source chunk is processed."
        ),
    )
    parser.add_argument(
        "--pass-through-limit",
        type=int,
        default=2000,
        help=(
            "Train split limit for speed-stressed examples whose faithful "
            "translation already fits the S2S RTF budget."
        ),
    )
    parser.add_argument("--dev-pass-through-limit", type=int, default=300)
    parser.add_argument("--test-pass-through-limit", type=int, default=300)
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
    parser.add_argument(
        "--tts-backend",
        default="qwen3_tts",
        choices=["qwen3_tts", "moss_tts", "higgs_tts"],
        help="TTS backend metadata to attach to sampled policy records.",
    )
    parser.add_argument(
        "--tts-model-path",
        default=None,
        help="Optional TTS model override stored in sampled policy metadata.",
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


def target_unit_rate_name(lang: str) -> str:
    return "zh_char_per_sec" if lang.startswith("zh") else "word_per_sec"


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
    max_target_duration_s: float | None = None,
    metadata_updates: dict[str, Any] | None = None,
    force_speed_suffix: bool = False,
    policy_sample_kind: str | None = None,
    tts_metadata: dict[str, Any] | None = None,
) -> S2SSample:
    src = clean_text(row["src_text"])
    tgt = clean_text(row["tgt_text"])
    tgt_lang = row.get("tgt_lang", "zh")
    scaled_duration = duration_s / speed_factor
    sample_id = (
        row["id"]
        if mode == "faithful" and speed_factor == 1.0 and not force_speed_suffix
        else f"{row['id']}__speed_{speed_factor:g}"
    )
    trajectory = parse_trajectory(row.get("trajectory", ""))
    max_target_chars = max_target_units if tgt_lang.startswith("zh") else None
    max_target_words = None if tgt_lang.startswith("zh") else max_target_units
    target = CompressionTarget(
        mode=mode,
        target_ratio=target_ratio,
        max_target_chars=max_target_chars,
        max_target_words=max_target_words,
        max_target_duration_s=round(
            max_target_duration_s if max_target_duration_s is not None else scaled_duration + max_end_lag_s,
            3,
        ),
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
            **({"policy_sample_kind": policy_sample_kind} if policy_sample_kind else {}),
            **(tts_metadata or {}),
            **(metadata_updates or {}),
        },
    )


def s2s_rtf_decision(
    *,
    duration_s: float,
    speed_factor: float,
    target_units: int,
    target_unit_rate_value: float,
    rtf_threshold: float,
    rtf_lag_slack_s: float,
) -> dict[str, Any]:
    source_wall_duration_s = duration_s / speed_factor
    playback_budget_s = max(1e-6, source_wall_duration_s + rtf_lag_slack_s)
    allowed_speech_s = max(1e-6, playback_budget_s * rtf_threshold)
    max_target_units = max(1, int(allowed_speech_s * target_unit_rate_value))
    reference_tts_duration_s = target_units / target_unit_rate_value
    s2s_rtf = reference_tts_duration_s / playback_budget_s
    return {
        "source_wall_duration_s": round(source_wall_duration_s, 3),
        "playback_budget_s": round(playback_budget_s, 3),
        "allowed_speech_s": round(allowed_speech_s, 3),
        "reference_tts_duration_s": round(reference_tts_duration_s, 3),
        "default_target_unit_rate": target_unit_rate_value,
        "rtf_threshold": rtf_threshold,
        "rtf_lag_slack_s": rtf_lag_slack_s,
        "max_target_units": max_target_units,
        "s2s_rtf": round(s2s_rtf, 6),
        "needs_compression": target_units > max_target_units,
    }


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
    if limit < 0:
        items.append(item)
        return
    if limit == 0:
        return
    if len(items) < limit:
        items.append(item)
        return
    j = rng.randint(0, seen - 1)
    if j < limit:
        items[j] = item


def speed_key(speed: float) -> str:
    return f"{speed:g}"


def empty_rtf_stats(speed_factors: list[float]) -> dict[str, Any]:
    return {
        "overall": empty_decision_stats(),
        "by_speed": {speed_key(speed): empty_decision_stats() for speed in speed_factors},
    }


def empty_decision_stats() -> dict[str, Any]:
    return {
        "total": 0,
        "pass_through": 0,
        "needs_compression": 0,
        "rtf_sum": 0.0,
        "rtf_min": None,
        "rtf_max": None,
        "budget_ratio_sum": 0.0,
        "budget_ratio_min": None,
        "budget_ratio_max": None,
        "rtf_histogram": {rtf_bin_label(i): 0 for i in range(len(RTF_BIN_EDGES) + 1)},
    }


def rtf_bin_label(index: int) -> str:
    if index == 0:
        return f"<= {RTF_BIN_EDGES[0]:g}"
    if index == len(RTF_BIN_EDGES):
        return f"> {RTF_BIN_EDGES[-1]:g}"
    return f"({RTF_BIN_EDGES[index - 1]:g}, {RTF_BIN_EDGES[index]:g}]"


def rtf_bin_index(rtf: float) -> int:
    for index, edge in enumerate(RTF_BIN_EDGES):
        if rtf <= edge:
            return index
    return len(RTF_BIN_EDGES)


def update_decision_stats(stats: dict[str, Any], decision: dict[str, Any], target_units: int) -> None:
    rtf = float(decision["s2s_rtf"])
    budget_ratio = float(decision["max_target_units"]) / max(1, target_units)
    stats["total"] += 1
    if decision["needs_compression"]:
        stats["needs_compression"] += 1
    else:
        stats["pass_through"] += 1
    stats["rtf_sum"] += rtf
    stats["rtf_min"] = rtf if stats["rtf_min"] is None else min(stats["rtf_min"], rtf)
    stats["rtf_max"] = rtf if stats["rtf_max"] is None else max(stats["rtf_max"], rtf)
    stats["budget_ratio_sum"] += budget_ratio
    stats["budget_ratio_min"] = (
        budget_ratio
        if stats["budget_ratio_min"] is None
        else min(stats["budget_ratio_min"], budget_ratio)
    )
    stats["budget_ratio_max"] = (
        budget_ratio
        if stats["budget_ratio_max"] is None
        else max(stats["budget_ratio_max"], budget_ratio)
    )
    stats["rtf_histogram"][rtf_bin_label(rtf_bin_index(rtf))] += 1


def update_rtf_stats(
    rtf_stats: dict[str, Any],
    speed: float,
    decision: dict[str, Any],
    target_units: int,
) -> None:
    update_decision_stats(rtf_stats["overall"], decision, target_units)
    update_decision_stats(rtf_stats["by_speed"][speed_key(speed)], decision, target_units)


def finalize_decision_stats(stats: dict[str, Any]) -> dict[str, Any]:
    total = stats["total"]
    if not total:
        return {
            "total": 0,
            "pass_through": 0,
            "needs_compression": 0,
            "pass_through_rate": None,
            "compression_rate": None,
            "rtf_mean": None,
            "rtf_min": None,
            "rtf_max": None,
            "budget_ratio_mean": None,
            "budget_ratio_min": None,
            "budget_ratio_max": None,
            "rtf_histogram": dict(stats["rtf_histogram"]),
        }
    return {
        "total": total,
        "pass_through": stats["pass_through"],
        "needs_compression": stats["needs_compression"],
        "pass_through_rate": round(stats["pass_through"] / total, 6),
        "compression_rate": round(stats["needs_compression"] / total, 6),
        "rtf_mean": round(stats["rtf_sum"] / total, 6),
        "rtf_min": stats["rtf_min"],
        "rtf_max": stats["rtf_max"],
        "budget_ratio_mean": round(stats["budget_ratio_sum"] / total, 6),
        "budget_ratio_min": round(stats["budget_ratio_min"], 6),
        "budget_ratio_max": round(stats["budget_ratio_max"], 6),
        "rtf_histogram": dict(stats["rtf_histogram"]),
    }


def finalize_rtf_stats(rtf_stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall": finalize_decision_stats(rtf_stats["overall"]),
        "by_speed": {
            speed: finalize_decision_stats(stats)
            for speed, stats in rtf_stats["by_speed"].items()
        },
    }


def empty_split_bucket(speed_factors: list[float]) -> dict[str, Any]:
    return {
        "faithful": [],
        "pass_through": [],
        "teacher": [],
        "policy": [],
        "seen_faithful": 0,
        "seen_pass_through": 0,
        "seen_teacher": 0,
        "seen_policy": 0,
        "usable_rows": 0,
        "base_ids": set(),
        "rtf_stats": empty_rtf_stats(speed_factors),
    }


def write_split_outputs(
    out_dir: Path,
    split_name: str,
    faithful: list[S2SSample],
    pass_through: list[S2SSample],
    teacher: list[S2SSample],
    policy: list[S2SSample],
) -> dict[str, str]:
    split_dir = out_dir / "splits" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    faithful_manifest = split_dir / "faithful_manifest.jsonl"
    faithful_sft = split_dir / "faithful_sft.jsonl"
    pass_through_manifest = split_dir / "pass_through_manifest.jsonl"
    pass_through_sft = split_dir / "pass_through_sft.jsonl"
    policy_sample_manifest = split_dir / "policy_sample_manifest.jsonl"
    rtf_decision_manifest = split_dir / "rtf_decision_manifest.jsonl"
    teacher_manifest = split_dir / "compression_teacher_manifest.jsonl"
    teacher_requests = split_dir / "compression_teacher_requests.jsonl"

    write_jsonl(faithful_manifest, (sample.to_dict() for sample in faithful))
    write_jsonl(faithful_sft, (sft_record(sample) for sample in faithful))
    write_jsonl(pass_through_manifest, (sample.to_dict() for sample in pass_through))
    write_jsonl(pass_through_sft, (sft_record(sample) for sample in pass_through))
    write_jsonl(policy_sample_manifest, (sample.to_dict() for sample in policy))
    write_jsonl(
        rtf_decision_manifest,
        (sample.to_dict() for sample in [*pass_through, *teacher]),
    )
    write_jsonl(teacher_manifest, (sample.to_dict() for sample in teacher))
    write_jsonl(teacher_requests, (teacher_record(sample) for sample in teacher))
    return {
        "faithful_manifest": str(faithful_manifest),
        "faithful_sft": str(faithful_sft),
        "pass_through_manifest": str(pass_through_manifest),
        "pass_through_sft": str(pass_through_sft),
        "policy_sample_manifest": str(policy_sample_manifest),
        "rtf_decision_manifest": str(rtf_decision_manifest),
        "teacher_manifest": str(teacher_manifest),
        "teacher_requests": str(teacher_requests),
    }


def write_train_alias_outputs(
    out_dir: Path,
    faithful: list[S2SSample],
    pass_through: list[S2SSample],
    teacher: list[S2SSample],
    policy: list[S2SSample],
) -> dict[str, str]:
    faithful_manifest = out_dir / "faithful_manifest.jsonl"
    faithful_sft = out_dir / "faithful_sft.jsonl"
    pass_through_manifest = out_dir / "pass_through_manifest.jsonl"
    pass_through_sft = out_dir / "pass_through_sft.jsonl"
    policy_sample_manifest = out_dir / "policy_sample_manifest.jsonl"
    rtf_decision_manifest = out_dir / "rtf_decision_manifest.jsonl"
    teacher_manifest = out_dir / "compression_teacher_manifest.jsonl"
    teacher_requests = out_dir / "compression_teacher_requests.jsonl"

    write_jsonl(faithful_manifest, (sample.to_dict() for sample in faithful))
    write_jsonl(faithful_sft, (sft_record(sample) for sample in faithful))
    write_jsonl(pass_through_manifest, (sample.to_dict() for sample in pass_through))
    write_jsonl(pass_through_sft, (sft_record(sample) for sample in pass_through))
    write_jsonl(policy_sample_manifest, (sample.to_dict() for sample in policy))
    write_jsonl(
        rtf_decision_manifest,
        (sample.to_dict() for sample in [*pass_through, *teacher]),
    )
    write_jsonl(teacher_manifest, (sample.to_dict() for sample in teacher))
    write_jsonl(teacher_requests, (teacher_record(sample) for sample in teacher))
    return {
        "faithful_manifest": str(faithful_manifest),
        "faithful_sft": str(faithful_sft),
        "pass_through_manifest": str(pass_through_manifest),
        "pass_through_sft": str(pass_through_sft),
        "policy_sample_manifest": str(policy_sample_manifest),
        "rtf_decision_manifest": str(rtf_decision_manifest),
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
    buckets = {name: empty_split_bucket(speed_factors) for name in split_names}
    global_rtf_stats = empty_rtf_stats(speed_factors)
    tts_metadata = tts_metadata_for_backend(args.tts_backend, args.tts_model_path)
    teacher_limits = {
        "train": args.teacher_limit,
        "dev": args.dev_teacher_limit,
        "test": args.test_teacher_limit,
    }
    pass_through_limits = {
        "train": args.pass_through_limit,
        "dev": args.dev_pass_through_limit,
        "test": args.test_pass_through_limit,
    }
    faithful_limits = {
        "train": args.faithful_limit,
        "dev": args.dev_faithful_limit,
        "test": args.test_faithful_limit,
    }
    policy_limits = {
        "train": args.policy_sample_size,
        "dev": args.dev_policy_sample_size,
        "test": args.test_policy_sample_size,
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

        if not args.summary_only and rng.random() < args.faithful_keep_prob:
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
            decision = s2s_rtf_decision(
                duration_s=duration_s,
                speed_factor=speed,
                target_units=tgt_units,
                target_unit_rate_value=rate,
                rtf_threshold=args.rtf_threshold,
                rtf_lag_slack_s=args.rtf_lag_slack_s,
            )
            update_rtf_stats(bucket["rtf_stats"], speed, decision, tgt_units)
            update_rtf_stats(global_rtf_stats, speed, decision, tgt_units)
            budget_units = int(decision["max_target_units"])
            ratio = budget_units / max(1, tgt_units)
            metadata_updates = {
                "streaming_decision_policy": "default_speech_s2s_rtf",
                "target_unit_rate_name": target_unit_rate_name(tgt_lang),
                **decision,
            }

            if not decision["needs_compression"]:
                bucket["seen_pass_through"] += 1
                if args.summary_only:
                    continue
                sample_for_policy = policy_limits.get(split_name, 0) > 0
                sample = make_sample_from_row(
                    row,
                    mode="faithful",
                    target_ratio=1.0,
                    max_target_units=budget_units,
                    speed_factor=speed,
                    duration_s=duration_s,
                    max_end_lag_s=args.max_end_lag_s,
                    max_target_duration_s=float(decision["allowed_speech_s"]),
                    metadata_updates=metadata_updates,
                    force_speed_suffix=True,
                    policy_sample_kind="pass_through" if sample_for_policy else None,
                    tts_metadata=tts_metadata if sample_for_policy else None,
                )
                if sample_for_policy:
                    bucket["seen_policy"] += 1
                    reservoir_add(
                        bucket["policy"],
                        sample,
                        policy_limits.get(split_name, 0),
                        bucket["seen_policy"],
                        rng,
                    )
                else:
                    reservoir_add(
                        bucket["pass_through"],
                        sample,
                        pass_through_limits.get(split_name, 0),
                        bucket["seen_pass_through"],
                        rng,
                    )
                continue

            bucket["seen_teacher"] += 1
            if args.summary_only:
                continue
            sample_for_policy = policy_limits.get(split_name, 0) > 0
            sample = make_sample_from_row(
                row,
                mode="concise" if ratio >= 0.62 else "very_concise",
                target_ratio=round(max(0.35, min(0.95, ratio)), 3),
                max_target_units=max(4, budget_units),
                speed_factor=speed,
                duration_s=duration_s,
                max_end_lag_s=args.max_end_lag_s,
                max_target_duration_s=float(decision["allowed_speech_s"]),
                metadata_updates=metadata_updates,
                policy_sample_kind="compression" if sample_for_policy else None,
                tts_metadata=tts_metadata if sample_for_policy else None,
            )
            if sample_for_policy:
                bucket["seen_policy"] += 1
                reservoir_add(
                    bucket["policy"],
                    sample,
                    policy_limits.get(split_name, 0),
                    bucket["seen_policy"],
                    rng,
                )
            else:
                reservoir_add(
                    bucket["teacher"],
                    sample,
                    teacher_limits.get(split_name, 0),
                    bucket["seen_teacher"],
                    rng,
                )

    for bucket in buckets.values():
        if bucket["policy"]:
            bucket["pass_through"] = [
                sample
                for sample in bucket["policy"]
                if sample.metadata.get("policy_sample_kind") == "pass_through"
            ]
            bucket["teacher"] = [
                sample
                for sample in bucket["policy"]
                if sample.metadata.get("policy_sample_kind") == "compression"
            ]

    summary_path = out_dir / "build_summary.json"
    split_outputs: dict[str, dict[str, str]] = {}
    if not args.summary_only:
        for split_name in split_names:
            split_outputs[split_name] = write_split_outputs(
                out_dir,
                split_name,
                buckets[split_name]["faithful"],
                buckets[split_name]["pass_through"],
                buckets[split_name]["teacher"],
                buckets[split_name]["policy"],
            )

    train_alias_outputs = {}
    if "train" in buckets and not args.summary_only:
        train_alias_outputs = write_train_alias_outputs(
            out_dir,
            buckets["train"]["faithful"],
            buckets["train"]["pass_through"],
            buckets["train"]["teacher"],
            buckets["train"]["policy"],
        )

    summary = {
        "tsv": args.tsv,
        "scanned_rows": scanned,
        "usable_rows": usable,
        "summary_only": args.summary_only,
        "split_seed": args.seed,
        "split_ratios": dict(split_ratios),
        "split_unit": "base_id",
        "split_limits": {
            "teacher": {name: teacher_limits.get(name, 0) for name in split_names},
            "pass_through": {
                name: pass_through_limits.get(name, 0) for name in split_names
            },
            "faithful": {name: faithful_limits.get(name, 0) for name in split_names},
            "policy_sample": {name: policy_limits.get(name, 0) for name in split_names},
        },
        "split_stats": {
            name: {
                "base_ids": len(bucket["base_ids"]),
                "usable_rows": bucket["usable_rows"],
                "faithful_records": len(bucket["faithful"]),
                "faithful_seen_candidates": bucket["seen_faithful"],
                "pass_through_records": len(bucket["pass_through"]),
                "pass_through_seen_candidates": bucket["seen_pass_through"],
                "teacher_records": len(bucket["teacher"]),
                "teacher_seen_candidates": bucket["seen_teacher"],
                "policy_records": len(bucket["policy"]),
                "policy_seen_candidates": bucket["seen_policy"],
                "policy_pass_through_records": sum(
                    1
                    for sample in bucket["policy"]
                    if sample.metadata.get("policy_sample_kind") == "pass_through"
                ),
                "policy_compression_records": sum(
                    1
                    for sample in bucket["policy"]
                    if sample.metadata.get("policy_sample_kind") == "compression"
                ),
                "rtf_decision_stats": finalize_rtf_stats(bucket["rtf_stats"]),
            }
            for name, bucket in buckets.items()
        },
        "natural_rtf_distribution": finalize_rtf_stats(global_rtf_stats),
        "base_id_overlap_check": pairwise_base_id_overlaps(buckets),
        "speed_factors": speed_factors,
        "compression_decision": {
            "policy": "default_speech_s2s_rtf",
            "rtf_threshold": args.rtf_threshold,
            "rtf_lag_slack_s": args.rtf_lag_slack_s,
            "zh_char_per_sec": args.zh_char_per_sec,
            "word_per_sec": args.word_per_sec,
            "legacy_stress_ratio_threshold_unused": args.stress_ratio_threshold,
        },
        "tts_backend": tts_metadata,
        "outputs": split_outputs,
        "train_alias_outputs": train_alias_outputs,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
