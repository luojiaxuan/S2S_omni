#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.hibiki_zero import SUPPORTED_SRC_LANGS, normalize_src_lang


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ST TSV rows into Hibiki-Zero source manifest JSONL."
    )
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--audio-root",
        default="",
        help="Joined with relative TSV audio paths. Defaults to the TSV parent.",
    )
    parser.add_argument("--max-per-lang", type=int, default=0)
    parser.add_argument("--require-supported", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunk-seconds", type=float, default=0.0)
    return parser.parse_args()


def detect_dialect(path: str | Path) -> csv.Dialect:
    sample = Path(path).read_text(encoding="utf-8", errors="replace")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,")
    except csv.Error:
        return csv.excel_tab


def audio_path(raw_audio: str, tsv_path: Path, audio_root: str) -> str:
    path = Path(raw_audio)
    if path.is_absolute():
        return str(path)
    root = Path(audio_root) if audio_root else tsv_path.parent
    return str((root / path).resolve())


def split_single_audio(audio: str, chunk_seconds: float) -> list[dict[str, Any]]:
    if chunk_seconds <= 0:
        return [{"chunk_index": 0, "source_audio": audio}]
    return [
        {
            "chunk_index": 0,
            "source_audio": audio,
            "metadata": {"chunking": "single_audio_placeholder", "chunk_seconds": chunk_seconds},
        }
    ]


def convert_row(row: dict[str, str], tsv_path: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    src_lang_raw = row.get("src_lang") or row.get("source_language") or ""
    tgt_lang = (row.get("tgt_lang") or row.get("target_language") or "en").lower()
    try:
        src_lang = normalize_src_lang(src_lang_raw)
    except ValueError:
        if args.require_supported:
            return None
        src_lang = src_lang_raw.lower()
    if args.require_supported and src_lang not in SUPPORTED_SRC_LANGS:
        return None
    if tgt_lang != "en":
        return None
    raw_audio = row.get("audio") or row.get("audio_path") or row.get("source_audio") or ""
    if not raw_audio:
        return None
    sample_id = row.get("id") or row.get("sample_id") or Path(raw_audio).stem
    resolved_audio = audio_path(raw_audio, tsv_path, args.audio_root)
    return {
        "sample_id": sample_id,
        "src_lang": src_lang,
        "source_text": row.get("src_text") or row.get("source_text") or "",
        "reference_en_text": row.get("tgt_text") or row.get("target_text") or "",
        "source_audio_chunks": split_single_audio(resolved_audio, args.chunk_seconds),
        "metadata": {
            "source_tsv": str(tsv_path),
            "speaker": row.get("speaker") or "",
            "n_frames": row.get("n_frames") or "",
        },
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    written = 0
    skipped = 0
    with output.open("w", encoding="utf-8") as out:
        for input_path in args.input:
            path = Path(input_path)
            dialect = detect_dialect(path)
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.DictReader(handle, dialect=dialect)
                for row in reader:
                    converted = convert_row(row, path, args)
                    if converted is None:
                        skipped += 1
                        continue
                    lang = str(converted["src_lang"])
                    if args.max_per_lang > 0 and counts.get(lang, 0) >= args.max_per_lang:
                        skipped += 1
                        continue
                    out.write(json.dumps(converted, ensure_ascii=False, sort_keys=False) + "\n")
                    written += 1
                    counts[lang] = counts.get(lang, 0) + 1
    summary = {
        "inputs": args.input,
        "output": str(output),
        "written": written,
        "skipped": skipped,
        "counts_by_lang": dict(sorted(counts.items())),
        "require_supported": args.require_supported,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
