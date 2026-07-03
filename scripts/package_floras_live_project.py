#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


RUNS = {
    "openai_chunk960": "openai_eval_full_enzh_chunk960_asr",
    "openai_chunk1920": "openai_eval_full_enzh_chunk1920_asr",
    "gemini_chunk960": "gemini_eval_full_enzh_chunk960_trim_asr",
    "gemini_chunk1920": "gemini_eval_full_enzh_chunk1920_trim_asr",
}

ROOT_FILES = [
    "live_runs.jsonl",
    "selected_samples.jsonl",
    "manifest_meta.json",
    "openai_asr_full_enzh_chunk960.jsonl",
    "openai_asr_full_enzh_chunk1920.jsonl",
    "gemini_asr_full_enzh_chunk960_trim.jsonl",
    "gemini_asr_full_enzh_chunk1920_trim.jsonl",
    "openai_window_asr_full_enzh_chunk960_speed1.jsonl",
    "openai_mfa_target_windows_enzh_chunk960_speed1.jsonl",
]

RUN_FILES = [
    "summary.json",
    "metrics.jsonl",
    "timeline.jsonl",
    "sentence_coverage.jsonl",
    "index.html",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package lightweight FLORAS live benchmark artifacts.")
    parser.add_argument("--source-dir", required=True, help="Source FLORAS output directory.")
    parser.add_argument("--project-dir", required=True, help="Destination project directory inside the repo.")
    return parser.parse_args()


def copy_file(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"path": str(dst), "bytes": dst.stat().st_size}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_manifest(project_dir: Path, source_dir: Path, copied: list[dict[str, Any]]) -> None:
    large_audio = []
    for pattern in ("*.wav", "*.mp3", "*.flac"):
        for path in source_dir.glob(f"**/{pattern}"):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            large_audio.append({"path": str(path), "bytes": size})
    large_audio.sort(key=lambda row: row["bytes"], reverse=True)
    copied_rel = []
    for row in copied:
        try:
            rel_path = str(Path(row["path"]).resolve().relative_to(project_dir))
        except ValueError:
            rel_path = str(row["path"])
        copied_rel.append({"path": rel_path, "bytes": row["bytes"]})
    manifest = {
        "name": "floras_live_s2s_benchmark",
        "source_dir": str(source_dir),
        "copied_files": copied_rel,
        "large_audio_policy": "Audio and per-window wav files are not checked into Git; use source_dir or regenerate.",
        "large_audio_examples": large_audio[:40],
    }
    (project_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_summary_table(project_dir: Path) -> None:
    metrics_path = project_dir / "artifacts" / "compare_openai_gemini_enzh_full_chunks" / "compare_metrics.jsonl"
    rows = read_jsonl(metrics_path)
    table = [
        "| run | backend | chunk_ms | speed | BLEU | chrF | CER | duration_lag_s | wall_delay_s | max_backlog_s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table.append(
            "| {run_id} | {backend} | {chunk_ms} | {speed:.2f} | {bleu:.2f} | {chrf:.2f} | {cer:.3f} | {lag:.2f} | {wall:.2f} | {backlog:.2f} |".format(
                run_id=row.get("run_id", ""),
                backend=row.get("compare_backend", ""),
                chunk_ms=int(row.get("compare_chunk_ms") or 0),
                speed=float(row.get("speed_factor") or 0.0),
                bleu=float(row.get("bleu") or 0.0),
                chrf=float(row.get("chrf") or 0.0),
                cer=float(row.get("cer") or 0.0),
                lag=float(row.get("duration_end_lag_s") or row.get("end_lag_s") or 0.0),
                wall=float(row.get("wall_clock_end_delay_s") or 0.0),
                backlog=float(row.get("max_backlog_s") or 0.0),
            )
        )
    (project_dir / "RESULTS.md").write_text("\n".join(table) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    project_dir = Path(args.project_dir).resolve()
    artifacts_dir = project_dir / "artifacts"
    copied: list[dict[str, Any]] = []

    compare_src = source_dir / "compare_openai_gemini_enzh_full_chunks"
    compare_dst = artifacts_dir / "compare_openai_gemini_enzh_full_chunks"
    for filename in ("index.html", "compare_metrics.jsonl"):
        copied.append(copy_file(compare_src / filename, compare_dst / filename))

    root_dst = artifacts_dir / "root_metadata"
    for filename in ROOT_FILES:
        src = source_dir / filename
        if src.exists():
            copied.append(copy_file(src, root_dst / filename))

    eval_dst = artifacts_dir / "eval_runs"
    for label, dirname in RUNS.items():
        run_src = source_dir / dirname
        for filename in RUN_FILES:
            src = run_src / filename
            if src.exists():
                copied.append(copy_file(src, eval_dst / label / filename))

    write_manifest(project_dir, source_dir, copied)
    write_summary_table(project_dir)
    print(json.dumps({"project_dir": str(project_dir), "copied_files": len(copied)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
