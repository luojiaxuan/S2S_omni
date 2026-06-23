#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.codec_data import base_id_from_id
from s2s_omni.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build split-isolated GigaSpeech manifests for Omni wav/code pair generation."
    )
    parser.add_argument(
        "--tsv",
        default="/mnt/taurus/data/siqiouyang/datasets/gigaspeech/train_xl_case_ft-qwen2.5-32b-instruct_marked_mfa_punc_asr.tsv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-counts", default="train=25000,dev=500,test=500")
    parser.add_argument("--split-ratios", default="train=0.9,dev=0.05,test=0.05")
    parser.add_argument("--seed", type=int, default=260623)
    parser.add_argument("--src-lang", default="en")
    parser.add_argument("--tgt-lang", default="zh")
    parser.add_argument("--span-unit-rate", type=float, default=16000.0)
    parser.add_argument("--min-duration-s", type=float, default=1.0)
    parser.add_argument("--max-duration-s", type=float, default=30.0)
    parser.add_argument("--max-scan-rows", type=int, default=0)
    parser.add_argument("--check-audio-exists", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-every", type=int, default=50000)
    return parser.parse_args()


def parse_key_values(spec: str, value_type: type) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid key/value item: {item!r}")
        key, raw = item.split("=", 1)
        out[key.strip()] = value_type(raw)
    return out


def parse_audio_span(uri: str) -> tuple[str, int | None, int | None]:
    parts = uri.rsplit(":", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0], int(parts[1]), int(parts[2])
    return uri, None, None


def stable_unit_interval(value: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / float(2**64)


def split_for_base_id(base_id: str, ratios: dict[str, float], seed: int) -> str:
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    point = stable_unit_interval(base_id, seed) * total
    cumulative = 0.0
    last_name = ""
    for name, ratio in ratios.items():
        last_name = name
        cumulative += ratio
        if point < cumulative:
            return name
    return last_name


def row_to_manifest_record(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any] | None:
    if row.get("src_lang") != args.src_lang or row.get("tgt_lang") != args.tgt_lang:
        return None
    source_text = (row.get("src_text") or "").strip()
    target_text = (row.get("tgt_text") or "").strip()
    if not source_text or not target_text:
        return None
    audio_uri = row.get("audio") or ""
    audio_path, _offset, span_frames = parse_audio_span(audio_uri)
    if not audio_path or span_frames is None:
        return None
    if args.check_audio_exists and not Path(audio_path).exists():
        return None
    duration_s = span_frames / float(args.span_unit_rate)
    if duration_s < args.min_duration_s or duration_s > args.max_duration_s:
        return None
    sample_id = str(row.get("id") or "")
    if not sample_id:
        return None
    base_id = base_id_from_id(sample_id)
    return {
        "id": sample_id,
        "base_id": base_id,
        "audio_path": audio_uri,
        "source_audio": audio_uri,
        "source_text": source_text,
        "reference_translation": target_text,
        "src_lang": row.get("src_lang", args.src_lang),
        "tgt_lang": row.get("tgt_lang", args.tgt_lang),
        "duration_s": round(duration_s, 6),
        "n_frames": int(row.get("n_frames") or span_frames),
        "speaker": row.get("speaker"),
        "metadata": {
            "source_tsv": str(args.tsv),
            "original_audio_path": audio_path,
            "span_unit_rate": args.span_unit_rate,
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {key: int(value) for key, value in parse_key_values(args.target_counts, int).items()}
    ratios = {key: float(value) for key, value in parse_key_values(args.split_ratios, float).items()}
    if set(targets) - set(ratios):
        raise SystemExit("--target-counts contains splits not present in --split-ratios")

    rows_by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in targets}
    used_base_ids: set[str] = set()
    scanned = 0
    eligible = 0
    csv.field_size_limit(sys.maxsize)
    with Path(args.tsv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for scanned, row in enumerate(reader, start=1):
            if args.max_scan_rows > 0 and scanned > args.max_scan_rows:
                break
            record = row_to_manifest_record(row, args)
            if record is None:
                continue
            eligible += 1
            base_id = record["base_id"]
            if base_id in used_base_ids:
                continue
            split = split_for_base_id(base_id, ratios, args.seed)
            if split not in rows_by_split or len(rows_by_split[split]) >= targets[split]:
                continue
            record["split"] = split
            rows_by_split[split].append(record)
            used_base_ids.add(base_id)
            if args.log_every > 0 and scanned % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "scanned": scanned,
                            "eligible": eligible,
                            "counts": {k: len(v) for k, v in rows_by_split.items()},
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if all(len(rows_by_split[name]) >= target for name, target in targets.items()):
                break

    summary = {
        "tsv": args.tsv,
        "output_dir": str(output_dir),
        "seed": args.seed,
        "target_counts": targets,
        "split_ratios": ratios,
        "scanned_rows": scanned,
        "eligible_rows": eligible,
        "counts": {name: len(rows) for name, rows in rows_by_split.items()},
        "filled": {name: len(rows_by_split[name]) >= target for name, target in targets.items()},
    }
    for split, rows in rows_by_split.items():
        write_jsonl(output_dir / f"{split}_manifest.jsonl", rows)
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
