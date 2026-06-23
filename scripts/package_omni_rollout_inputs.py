#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.audio import load_audio_span  # noqa: E402
from s2s_omni.io import read_jsonl, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package GigaSpeech source spans as HF-ready wav rollout inputs."
    )
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--target-root", default="HF_DATASET_ROOT")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument(
        "--hf-repo-id",
        default="",
        help="Optional dataset repo id to upload the packaged folder to.",
    )
    parser.add_argument("--hf-private", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    keep = []
    for char in value:
        keep.append(char if char.isalnum() or char in "._-" else "_")
    return ("".join(keep)[:180] or "sample").strip("._") or "sample"


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32).reshape(-1), sample_rate, subtype="PCM_16")


def staged_audio_path(target_root: str, split: str, sample_id: str) -> str:
    return f"{target_root.rstrip('/')}/source_wav/{split}/{sanitize_name(sample_id)}.wav"


def local_audio_path(output_dir: Path, split: str, sample_id: str) -> Path:
    return output_dir / "source_wav" / split / f"{sanitize_name(sample_id)}.wav"


def package_records(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    manifest_out = output_dir / "manifests" / f"{args.split}_manifest.jsonl"
    rejected_out = output_dir / "manifests" / f"{args.split}_rejected.jsonl"
    done_ids: set[str] = set()
    if args.resume and manifest_out.exists():
        done_ids = {str(row.get("id")) for row in read_jsonl(manifest_out) if row.get("id")}

    records = read_jsonl(args.input_manifest)
    if args.max_records > 0:
        records = records[: args.max_records]

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    if args.resume and manifest_out.exists():
        accepted.extend(read_jsonl(manifest_out))

    for index, record in enumerate(records, start=1):
        sample_id = str(record.get("id") or "")
        if not sample_id:
            rejected.append({"source_record": record, "reject_reasons": ["missing_id"]})
            continue
        if sample_id in done_ids:
            continue
        source_audio = str(record.get("audio_path") or record.get("source_audio") or "")
        if not source_audio:
            rejected.append({"id": sample_id, "source_record": record, "reject_reasons": ["missing_audio"]})
            continue
        local_wav = local_audio_path(output_dir, args.split, sample_id)
        try:
            audio, sample_rate = load_audio_span(source_audio, target_sample_rate=args.sample_rate)
            write_wav(local_wav, audio, sample_rate)
            packaged_path = staged_audio_path(args.target_root, args.split, sample_id)
            row = dict(record)
            row["source_audio_original"] = source_audio
            row["audio_path"] = packaged_path
            row["source_audio"] = packaged_path
            row["packaged_source_wav"] = packaged_path
            row["packaged_sample_rate"] = sample_rate
            row["packaged_duration_s"] = round(float(audio.shape[0]) / float(sample_rate), 6)
            row["packaged_num_samples"] = int(audio.shape[0])
            accepted.append(row)
            done_ids.add(sample_id)
        except Exception as exc:  # noqa: BLE001
            rejected.append(
                {
                    "id": sample_id,
                    "source_record": record,
                    "reject_reasons": [f"exception:{type(exc).__name__}"],
                    "error": str(exc),
                }
            )
        if args.log_every > 0 and index % args.log_every == 0:
            print(
                json.dumps(
                    {"processed": index, "accepted": len(accepted), "rejected": len(rejected)},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    write_jsonl(manifest_out, accepted)
    if rejected:
        write_jsonl(rejected_out, rejected)

    summary = {
        "input_manifest": args.input_manifest,
        "output_dir": str(output_dir),
        "split": args.split,
        "target_root": args.target_root,
        "sample_rate": args.sample_rate,
        "selected": len(records),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "manifest": str(manifest_out),
        "rejected_manifest": str(rejected_out) if rejected else None,
    }
    summary_path = output_dir / "manifests" / f"{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def upload_to_hf(output_dir: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        path_in_repo=".",
    )


def main() -> None:
    args = parse_args()
    summary = package_records(args)
    if args.hf_repo_id:
        upload_to_hf(Path(args.output_dir), args.hf_repo_id, args.hf_private)
        summary["hf_repo_id"] = args.hf_repo_id
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
