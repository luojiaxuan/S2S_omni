#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable
from io import BytesIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.rasst import sanitize_id, write_mono_wav


LANG_CONFIGS = {
    "fr": "fr_fr",
    "es": "es_419",
    "pt": "pt_br",
    "de": "de_de",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export parallel FLEURS fr/es/pt/de speech + English reference text."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--languages", default="fr,es,pt,de")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--dataset", default="google/fleurs")
    parser.add_argument("--english-config", default="en_us")
    parser.add_argument("--max-per-lang", type=int, default=100)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def load_split(dataset_name: str, config: str, split: str, streaming: bool) -> Iterable[dict[str, Any]]:
    from datasets import Audio, load_dataset

    dataset = load_dataset(dataset_name, config, split=split, streaming=streaming)
    if hasattr(dataset, "cast_column") and "audio" in getattr(dataset, "column_names", []):
        dataset = dataset.cast_column("audio", Audio(decode=False))
    return dataset


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("sentence_id") or row.get("path") or "")


def transcription(row: dict[str, Any]) -> str:
    return str(
        row.get("transcription")
        or row.get("raw_transcription")
        or row.get("normalized_transcription")
        or ""
    )


def audio_array(row: dict[str, Any]) -> tuple[Any, int]:
    import soundfile as sf

    audio = row.get("audio")
    if not isinstance(audio, dict):
        raise ValueError("FLEURS row has no audio field")
    if audio.get("array") is not None:
        return audio["array"], int(audio.get("sampling_rate") or 16000)
    path_value = audio.get("path") or row.get("path")
    if path_value and Path(str(path_value)).exists():
        wav, sr = sf.read(str(path_value), dtype="float32", always_2d=False)
        return wav, int(sr)
    if audio.get("bytes"):
        wav, sr = sf.read(BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
        return wav, int(sr)
    raise ValueError("FLEURS row has neither decoded audio, path, nor bytes")


def build_english_refs(args: argparse.Namespace) -> tuple[dict[str, str], list[str]]:
    by_id = {}
    by_index = []
    for row in load_split(args.dataset, args.english_config, args.split, args.streaming):
        rid = row_id(row)
        text = transcription(row)
        if rid:
            by_id[rid] = text
        by_index.append(text)
    return by_id, by_index


def main() -> None:
    args = parse_args()
    requested = [item.strip() for item in args.languages.split(",") if item.strip()]
    configs = {lang: LANG_CONFIGS[lang] for lang in requested}
    output = Path(args.output)
    audio_dir = Path(args.audio_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    english_by_id, english_by_index = build_english_refs(args)
    written = 0
    skipped = 0
    counts: dict[str, int] = {}
    with output.open("w", encoding="utf-8") as handle:
        for lang, config in configs.items():
            counts[lang] = 0
            for row_index, row in enumerate(load_split(args.dataset, config, args.split, args.streaming)):
                if args.max_per_lang > 0 and counts[lang] >= args.max_per_lang:
                    break
                rid = row_id(row)
                reference = english_by_id.get(rid, "")
                if not reference and row_index < len(english_by_index):
                    reference = english_by_index[row_index]
                if not rid or not reference:
                    skipped += 1
                    continue
                try:
                    wav, sr = audio_array(row)
                    if sr != args.sample_rate:
                        from s2s_omni.audio import resample_audio

                        wav = resample_audio(wav, sr, args.sample_rate)
                        sr = args.sample_rate
                    sample_id = sanitize_id(f"fleurs_{lang}_{args.split}_{row_index:05d}_{rid}")
                    wav_path = audio_dir / lang / f"{sample_id}.wav"
                    write_mono_wav(wav_path, wav, sr)
                    record = {
                        "sample_id": sample_id,
                        "src_lang": lang,
                        "source_text": transcription(row),
                        "reference_en_text": reference,
                        "source_audio_chunks": [
                            {
                                "chunk_index": 0,
                                "source_audio": str(wav_path),
                                "source_text": transcription(row),
                                "reference_en_text": reference,
                            }
                        ],
                        "metadata": {
                            "dataset": args.dataset,
                            "split": args.split,
                            "fleurs_config": config,
                            "english_config": args.english_config,
                            "fleurs_id": rid,
                            "fleurs_row_index": row_index,
                        },
                    }
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")
                    written += 1
                    counts[lang] += 1
                except Exception:
                    skipped += 1
                    continue
    summary = {
        "output": str(output),
        "audio_dir": str(audio_dir),
        "dataset": args.dataset,
        "split": args.split,
        "written": written,
        "skipped": skipped,
        "counts_by_lang": counts,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
