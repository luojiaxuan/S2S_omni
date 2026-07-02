#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl, write_jsonl
from s2s_omni.textgrid import (
    choose_word_tier,
    nonsilence_intervals,
    normalize_label,
    parse_textgrid,
    transcript_for_mfa,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare/extract FLORAS target-side MFA alignment.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="Create MFA wav/txt corpus from an eval directory.")
    prep.add_argument("--eval-dir", required=True)
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--run-id", action="append", default=[])
    prep.add_argument("--copy-audio", action=argparse.BooleanOptionalAction, default=True)

    ext = sub.add_parser("extract", help="Convert MFA TextGrids into per-window aligned text JSONL.")
    ext.add_argument("--eval-dir", required=True)
    ext.add_argument("--mfa-work-dir", required=True)
    ext.add_argument("--textgrid-dir", default="")
    ext.add_argument("--output", required=True)
    ext.add_argument("--run-id", action="append", default=[])
    ext.add_argument("--overlap", choices=["midpoint", "any"], default="midpoint")

    prep_win = sub.add_parser("prepare-windows", help="Create MFA corpus from per-window target wavs/ASR.")
    prep_win.add_argument("--eval-dir", required=True)
    prep_win.add_argument("--window-asr-jsonl", required=True)
    prep_win.add_argument("--output-dir", required=True)
    prep_win.add_argument("--run-id", action="append", default=[])
    prep_win.add_argument("--copy-audio", action=argparse.BooleanOptionalAction, default=True)

    ext_win = sub.add_parser("extract-windows", help="Convert per-window TextGrids to target MFA JSONL.")
    ext_win.add_argument("--mfa-work-dir", required=True)
    ext_win.add_argument("--textgrid-dir", default="")
    ext_win.add_argument("--output", required=True)
    ext_win.add_argument("--run-id", action="append", default=[])
    return parser.parse_args()


def load_metrics(eval_dir: Path) -> list[dict[str, Any]]:
    metrics_path = eval_dir / "metrics.jsonl"
    rows = []
    if metrics_path.exists():
        rows.extend(read_jsonl(metrics_path))
    else:
        for path in sorted(eval_dir.glob("*/metrics.json")):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def safe_id(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return out.strip("_")[:180] or "sample"


def prepare(args: argparse.Namespace) -> None:
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.output_dir)
    corpus_dir = out_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    keep = set(args.run_id)
    manifest = []
    for row in load_metrics(eval_dir):
        run_id = str(row.get("run_id") or "")
        if keep and run_id not in keep:
            continue
        candidate_text = str(row.get("candidate_text") or "")
        mfa_text = transcript_for_mfa(candidate_text)
        wav_path = Path(str(row.get("generated_wav_path") or eval_dir / run_id / "generated_target.wav"))
        if not run_id or not mfa_text or not wav_path.exists():
            continue
        mfa_id = safe_id(run_id)
        out_wav = corpus_dir / f"{mfa_id}.wav"
        out_txt = corpus_dir / f"{mfa_id}.txt"
        if args.copy_audio:
            shutil.copy2(wav_path, out_wav)
        else:
            if out_wav.exists() or out_wav.is_symlink():
                out_wav.unlink()
            out_wav.symlink_to(wav_path)
        out_txt.write_text(mfa_text + "\n", encoding="utf-8")
        manifest.append(
            {
                "run_id": run_id,
                "mfa_id": mfa_id,
                "generated_wav_path": str(wav_path),
                "mfa_wav_path": str(out_wav),
                "mfa_txt_path": str(out_txt),
                "candidate_text_chars": len(candidate_text),
                "mfa_unit_count": len(mfa_text.split()),
            }
        )
    write_jsonl(out_dir / "manifest.jsonl", manifest)
    print(json.dumps({"rows": len(manifest), "corpus_dir": str(corpus_dir)}, ensure_ascii=False, indent=2))


def load_window_asr(path: str | Path) -> dict[tuple[str, int], str]:
    out = {}
    for row in read_jsonl(path):
        if row.get("run_id") is None or row.get("window_index") is None:
            continue
        out[(str(row["run_id"]), int(row["window_index"]))] = str(row.get("asr_text") or "").strip()
    return out


def prepare_windows(args: argparse.Namespace) -> None:
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.output_dir)
    corpus_dir = out_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    keep = set(args.run_id)
    window_asr = load_window_asr(args.window_asr_jsonl)
    manifest = []
    for timeline_path in sorted(eval_dir.glob("*/timeline.jsonl")):
        run_id = timeline_path.parent.name
        if keep and run_id not in keep:
            continue
        for window in read_jsonl(timeline_path):
            window_index = int(window["window_index"])
            asr_text = window_asr.get((run_id, window_index), "")
            mfa_text = transcript_for_mfa(asr_text)
            if not mfa_text:
                continue
            wav_path = Path(str(window.get("target_window_audio_path") or ""))
            if not wav_path.exists():
                wav_path = timeline_path.parent / str(window.get("target_window_audio_rel") or "")
            if not wav_path.exists():
                continue
            mfa_id = safe_id(f"{run_id}__window_{window_index:04d}")
            out_wav = corpus_dir / f"{mfa_id}.wav"
            out_txt = corpus_dir / f"{mfa_id}.txt"
            if args.copy_audio:
                shutil.copy2(wav_path, out_wav)
            else:
                if out_wav.exists() or out_wav.is_symlink():
                    out_wav.unlink()
                out_wav.symlink_to(wav_path)
            out_txt.write_text(mfa_text + "\n", encoding="utf-8")
            manifest.append(
                {
                    "run_id": run_id,
                    "window_index": window_index,
                    "mfa_id": mfa_id,
                    "target_window_audio_path": str(wav_path),
                    "target_audio_emitted_at_window_start_s": window.get("target_audio_emitted_at_window_start_s"),
                    "target_audio_emitted_until_boundary_s": window.get("target_audio_emitted_until_boundary_s"),
                    "asr_text": asr_text,
                    "mfa_unit_count": len(mfa_text.split()),
                }
            )
    write_jsonl(out_dir / "manifest.jsonl", manifest)
    print(json.dumps({"rows": len(manifest), "corpus_dir": str(corpus_dir)}, ensure_ascii=False, indent=2))


def find_textgrid(textgrid_dir: Path, mfa_id: str) -> Path | None:
    for name in (f"{mfa_id}.TextGrid", f"{mfa_id}.textgrid"):
        path = textgrid_dir / name
        if path.exists():
            return path
    matches = list(textgrid_dir.rglob(f"{mfa_id}.TextGrid")) + list(textgrid_dir.rglob(f"{mfa_id}.textgrid"))
    return matches[0] if matches else None


def intervals_for_window(intervals: list[Any], start_s: float, end_s: float, mode: str) -> list[Any]:
    out = []
    for interval in intervals:
        if mode == "midpoint":
            mid = (float(interval.start_s) + float(interval.end_s)) / 2.0
            hit = start_s <= mid < end_s
        else:
            hit = float(interval.end_s) > start_s and float(interval.start_s) < end_s
        if hit:
            out.append(interval)
    return out


def extract(args: argparse.Namespace) -> None:
    eval_dir = Path(args.eval_dir)
    work_dir = Path(args.mfa_work_dir)
    textgrid_dir = Path(args.textgrid_dir) if args.textgrid_dir else work_dir / "aligned"
    manifest = {str(row["run_id"]): row for row in read_jsonl(work_dir / "manifest.jsonl")}
    keep = set(args.run_id)
    rows = []
    for run_id, row in manifest.items():
        if keep and run_id not in keep:
            continue
        textgrid = find_textgrid(textgrid_dir, str(row["mfa_id"]))
        timeline_path = eval_dir / run_id / "timeline.jsonl"
        if textgrid is None or not timeline_path.exists():
            continue
        intervals = nonsilence_intervals(choose_word_tier(parse_textgrid(textgrid)))
        for window in read_jsonl(timeline_path):
            start_s = float(window.get("target_audio_emitted_at_window_start_s") or 0.0)
            end_s = float(window.get("target_audio_emitted_until_boundary_s") or 0.0)
            selected = intervals_for_window(intervals, start_s, end_s, args.overlap)
            text = "".join(normalize_label(interval.text) for interval in selected)
            rows.append(
                {
                    "run_id": run_id,
                    "window_index": int(window["window_index"]),
                    "target_mfa_asr_text": text,
                    "target_mfa_start_s": round(float(selected[0].start_s), 6) if selected else None,
                    "target_mfa_end_s": round(float(selected[-1].end_s), 6) if selected else None,
                    "target_mfa_unit_count": len(selected),
                    "textgrid_path": str(textgrid),
                }
            )
    write_jsonl(args.output, rows)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, ensure_ascii=False, indent=2))


def extract_windows(args: argparse.Namespace) -> None:
    work_dir = Path(args.mfa_work_dir)
    textgrid_dir = Path(args.textgrid_dir) if args.textgrid_dir else work_dir / "aligned"
    keep = set(args.run_id)
    rows = []
    for row in read_jsonl(work_dir / "manifest.jsonl"):
        run_id = str(row["run_id"])
        if keep and run_id not in keep:
            continue
        textgrid = find_textgrid(textgrid_dir, str(row["mfa_id"]))
        if textgrid is None:
            continue
        intervals = nonsilence_intervals(choose_word_tier(parse_textgrid(textgrid)))
        text = "".join(normalize_label(interval.text) for interval in intervals)
        window_start = float(row.get("target_audio_emitted_at_window_start_s") or 0.0)
        rows.append(
            {
                "run_id": run_id,
                "window_index": int(row["window_index"]),
                "target_mfa_asr_text": text,
                "target_mfa_start_s": round(window_start + float(intervals[0].start_s), 6)
                if intervals
                else None,
                "target_mfa_end_s": round(window_start + float(intervals[-1].end_s), 6)
                if intervals
                else None,
                "target_mfa_unit_count": len(intervals),
                "textgrid_path": str(textgrid),
            }
        )
    write_jsonl(args.output, rows)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "extract":
        extract(args)
    elif args.cmd == "prepare-windows":
        prepare_windows(args)
    elif args.cmd == "extract-windows":
        extract_windows(args)
    else:
        raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
