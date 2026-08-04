#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import wave
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TRAIN = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline.jsonl"
DEFAULT_DEV = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline_dev.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact InfiniSST/RASST en->zh package for MOSS-TTS-Realtime fine-tuning."
    )
    parser.add_argument("--train-jsonl", default=DEFAULT_TRAIN)
    parser.add_argument("--dev-jsonl", default=DEFAULT_DEV)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", default="infinisst-moss-tts-en-zh-segments-v1")
    parser.add_argument("--max-train-segments", type=int, default=0)
    parser.add_argument("--max-dev-segments", type=int, default=0)
    parser.add_argument("--compression", choices=["zstd", "gzip", "none"], default="zstd")
    parser.add_argument("--zstd-level", type=int, default=6)
    parser.add_argument("--skip-source-duration", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-audio-tars", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            count += 1
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
    return count


def normalize_target_text(value: Any) -> str:
    return "".join(str(value or "").split())


def audio_duration_s(path: str | Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            return float(frames) / float(rate) if rate else None
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_split(
    *,
    split: str,
    jsonl_path: str,
    max_segments: int,
    read_source_duration: bool,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    rows: list[dict[str, Any]] = []
    audio_entries: list[tuple[str, str]] = []
    seen_arc = set()
    stopped = False
    for row_index, record in enumerate(read_jsonl(jsonl_path)):
        audios = [str(item) for item in record.get("audios") or []]
        assistants = [
            message for message in record.get("messages") or []
            if message.get("role") == "assistant"
        ]
        for turn_index, source_audio in enumerate(audios):
            if turn_index >= len(assistants):
                continue
            target_text = normalize_target_text(assistants[turn_index].get("content"))
            if not target_text:
                continue
            source_path = Path(source_audio)
            if not source_path.exists():
                continue
            sample_id = f"{split}_r{row_index:06d}_t{turn_index:03d}"
            source_wav_rel = f"source_wavs/{split}/{sample_id}.wav"
            target_wav_rel = f"target_wavs/{split}/{sample_id}.wav"
            duration = audio_duration_s(source_path) if read_source_duration else None
            metadata = {
                "source_jsonl": jsonl_path,
                "source_row_index": row_index,
                "turn_index": turn_index,
                "merge_multiplier": record.get("merge_multiplier"),
                "original_source_audio": source_audio,
            }
            row = {
                "id": sample_id,
                "sample_id": sample_id,
                "split": split,
                "src_lang": "en",
                "tgt_lang": "zh",
                "source_wav": source_wav_rel,
                "ref_wav": source_wav_rel,
                "target_wav": target_wav_rel,
                "target_text": target_text,
                "source_duration_s": duration,
                "target_duration_s": None,
                "metadata": metadata,
            }
            rows.append(row)
            if source_wav_rel not in seen_arc:
                audio_entries.append((source_audio, source_wav_rel))
                seen_arc.add(source_wav_rel)
            if max_segments > 0 and len(rows) >= max_segments:
                stopped = True
                break
        if stopped:
            break
    return rows, audio_entries


def moss_request_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "split": row["split"],
        "input": row["target_text"],
        "target_text": row["target_text"],
        "ref_wav": row["ref_wav"],
        "source_wav": row["source_wav"],
        "output_wav": row["target_wav"],
        "metadata": row["metadata"],
    }


def moss_raw_unresolved_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "ref_wav": row["ref_wav"],
        "conversations": [
            {
                "role": "assistant",
                "text": row["target_text"],
                "wav": row["target_wav"],
            }
        ],
        "metadata": {
            "split": row["split"],
            "src_lang": row["src_lang"],
            "tgt_lang": row["tgt_lang"],
            **row["metadata"],
        },
    }


def add_audio_to_tar(tar: tarfile.TarFile, entries: list[tuple[str, str]]) -> None:
    for source, arcname in entries:
        tar.add(source, arcname=arcname, recursive=False)


def write_tar(entries: list[tuple[str, str]], output_path: Path, compression: str, zstd_level: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if compression == "zstd" and shutil.which("zstd"):
        final_path = output_path.with_suffix(output_path.suffix + ".zst") if output_path.suffix != ".zst" else output_path
        proc = subprocess.Popen(
            ["zstd", "-T0", f"-{zstd_level}", "-q", "-o", str(final_path)],
            stdin=subprocess.PIPE,
        )
        if proc.stdin is None:
            raise RuntimeError("failed to open zstd stdin")
        try:
            with tarfile.open(fileobj=proc.stdin, mode="w|") as tar:
                add_audio_to_tar(tar, entries)
        finally:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        if proc.wait() != 0:
            raise RuntimeError(f"zstd failed while writing {final_path}")
        return final_path
    if compression == "gzip":
        final_path = output_path.with_suffix(output_path.suffix + ".gz") if output_path.suffix != ".gz" else output_path
        with tarfile.open(final_path, mode="w:gz") as tar:
            add_audio_to_tar(tar, entries)
        return final_path
    final_path = output_path
    with tarfile.open(final_path, mode="w") as tar:
        add_audio_to_tar(tar, entries)
    return final_path


def write_readme(root: Path, dataset_id: str, summary: dict[str, Any]) -> None:
    text = f"""# {dataset_id}

This package contains InfiniSST/RASST en->zh translation segments prepared for
MOSS-TTS-Realtime fine-tuning.

Files:

- `manifest/train_segments.jsonl`, `manifest/dev_segments.jsonl`: rich segment manifest.
- `manifest/*_moss_requests.jsonl`: input for `scripts/generate_moss_realtime_targets.py`.
- `manifest/*_moss_raw_unresolved.jsonl`: MOSS finetuning raw JSONL shape before target wav generation.
- `audio/*_source_wavs.tar*`: source/reference wav archives; extract at this package root.

Target wavs are intentionally not included here. Generate them with
MOSS-TTS-Realtime serving on the training host, then run upstream
`moss_tts_realtime/finetuning/prepare_data.py`.

Summary:

```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir) / args.dataset_id
    manifest_dir = root / "manifest"
    audio_dir = root / "audio"
    root.mkdir(parents=True, exist_ok=True)

    train_rows, train_audio = collect_split(
        split="train",
        jsonl_path=args.train_jsonl,
        max_segments=args.max_train_segments,
        read_source_duration=not args.skip_source_duration,
    )
    dev_rows, dev_audio = collect_split(
        split="dev",
        jsonl_path=args.dev_jsonl,
        max_segments=args.max_dev_segments,
        read_source_duration=not args.skip_source_duration,
    )

    write_jsonl(manifest_dir / "train_segments.jsonl", train_rows)
    write_jsonl(manifest_dir / "dev_segments.jsonl", dev_rows)
    write_jsonl(manifest_dir / "train_moss_requests.jsonl", (moss_request_row(row) for row in train_rows))
    write_jsonl(manifest_dir / "dev_moss_requests.jsonl", (moss_request_row(row) for row in dev_rows))
    write_jsonl(
        manifest_dir / "train_moss_raw_unresolved.jsonl",
        (moss_raw_unresolved_row(row) for row in train_rows),
    )
    write_jsonl(
        manifest_dir / "dev_moss_raw_unresolved.jsonl",
        (moss_raw_unresolved_row(row) for row in dev_rows),
    )

    artifacts: list[dict[str, Any]] = []
    if not args.skip_audio_tars:
        for split, entries in [("train", train_audio), ("dev", dev_audio)]:
            tar_path = write_tar(
                entries,
                audio_dir / f"{split}_source_wavs.tar",
                compression=args.compression,
                zstd_level=args.zstd_level,
            )
            artifacts.append(
                {
                    "path": str(tar_path.relative_to(root)),
                    "bytes": tar_path.stat().st_size,
                    "sha256": sha256_file(tar_path),
                    "entries": len(entries),
                }
            )

    summary = {
        "dataset_id": args.dataset_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "train_jsonl": args.train_jsonl,
        "dev_jsonl": args.dev_jsonl,
        "train_segments": len(train_rows),
        "dev_segments": len(dev_rows),
        "train_audio_entries": len(train_audio),
        "dev_audio_entries": len(dev_audio),
        "compression": args.compression,
        "source_duration_read": not args.skip_source_duration,
        "audio_artifacts": artifacts,
    }
    (root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_readme(root, args.dataset_id, summary)
    print(json.dumps({"output_dir": str(root), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
