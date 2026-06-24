#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl
from s2s_omni.rasst import load_mono_audio, sanitize_id, write_mono_wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hibiki-Zero generate baseline on a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hibiki-bin", default="hibiki-zero")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])
    return parser.parse_args()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
        handle.flush()


def source_concat_path(row: dict[str, Any], output_dir: Path) -> Path:
    sample_id = sanitize_id(str(row.get("sample_id") or row.get("id")))
    return output_dir / "source_concat" / f"{sample_id}.wav"


def ensure_source_concat(row: dict[str, Any], output_dir: Path, sample_rate: int = 16000) -> Path:
    out = source_concat_path(row, output_dir)
    if out.exists():
        return out
    chunks = row.get("source_audio_chunks") or []
    if not chunks:
        audio = row.get("source_audio") or row.get("audio")
        if not audio:
            raise ValueError("row has no source audio")
        chunks = [{"source_audio": audio}]
    wavs = []
    for chunk in chunks:
        wav, _ = load_mono_audio(chunk.get("source_audio"), sample_rate)
        wavs.append(wav)
    import numpy as np

    write_mono_wav(out, np.concatenate(wavs) if wavs else np.zeros(1, dtype=np.float32), sample_rate)
    return out


def snapshot_wavs(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path.resolve() for path in root.rglob("*.wav")}


def select_generated_wav(before: set[Path], root: Path) -> Path | None:
    after = snapshot_wavs(root)
    new_paths = sorted(after - before, key=lambda path: path.stat().st_mtime, reverse=True)
    if not new_paths:
        return None
    mono_paths = [path for path in new_paths if path.name.endswith("_mono.wav")]
    if mono_paths:
        return mono_paths[0]
    non_stereo_paths = [path for path in new_paths if not path.name.endswith("_stereo.wav")]
    if non_stereo_paths:
        return non_stereo_paths[0]
    return new_paths[0]


def run_one(row: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    sample_id = sanitize_id(str(row.get("sample_id") or row.get("id")))
    sample_dir = output_dir / "baseline_raw" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    source_wav = ensure_source_concat(row, output_dir)
    command = [args.hibiki_bin, "generate", "--file", str(source_wav), *args.extra_arg]
    record = {
        "sample_id": sample_id,
        "src_lang": row.get("src_lang"),
        "source_wav": str(source_wav),
        "command": command,
    }
    if args.dry_run:
        record.update({"accepted": True, "dry_run": True})
        return record
    before = snapshot_wavs(sample_dir)
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=sample_dir,
        text=True,
        capture_output=True,
        timeout=args.timeout_s,
        check=False,
    )
    elapsed = round(time.perf_counter() - started, 6)
    wav_path = select_generated_wav(before, sample_dir)
    stable_wav_path = ""
    if wav_path:
        stable = output_dir / "baseline_wav" / f"{sample_id}.wav"
        stable.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav_path, stable)
        stable_wav_path = str(stable)
    record.update(
        {
            "accepted": proc.returncode == 0 and bool(stable_wav_path),
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
            "generated_wav_path": stable_wav_path,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    )
    if not record["accepted"]:
        record["reject_reasons"] = ["no_generated_wav" if proc.returncode == 0 else "nonzero_exit"]
    return record


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_path = output_dir / "baseline_predictions.jsonl"
    if output_path.exists():
        output_path.unlink()
    accepted = rejected = 0
    for idx, row in enumerate(read_jsonl(args.manifest)):
        if args.max_records and idx >= args.max_records:
            break
        try:
            record = run_one(row, args, output_dir)
        except Exception as exc:
            record = {
                "sample_id": row.get("sample_id") or row.get("id"),
                "accepted": False,
                "reject_reasons": [f"exception:{type(exc).__name__}"],
                "error": str(exc),
            }
        append_jsonl(output_path, record)
        if record.get("accepted"):
            accepted += 1
        else:
            rejected += 1
        print(json.dumps({"processed": idx + 1, "accepted": accepted, "rejected": rejected}), flush=True)
    summary = {
        "manifest": args.manifest,
        "output": str(output_path),
        "accepted": accepted,
        "rejected": rejected,
        "hibiki_bin": args.hibiki_bin,
        "dry_run": args.dry_run,
    }
    (output_dir / "baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
