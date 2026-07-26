#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

LANGUAGES = (
    ("zh", "En-Zh", "Chinese", "enzh"),
    ("de", "En-De", "German", "ende"),
    ("ja", "En-Ja", "Japanese", "enja"),
)
SPEEDS = (
    (1.0, "1x", "1", "speed1"),
    (1.25, "1.25x", "1p25", "speed1p25"),
    (1.5, "1.5x", "1p5", "speed1p5"),
)
SAMPLE_ID = "2022.acl-long.117"
SAMPLE_DIR = f"004_{SAMPLE_ID}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the ACL6060 multilingual source-speed listening sample."
    )
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preview-duration-s", type=float, default=90.0)
    return parser.parse_args()


def duration_s(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_mp3(source: Path, destination: Path, preview_duration_s: float) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-t",
            f"{preview_duration_s:g}",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-b:a",
            "64k",
            str(destination),
        ],
        check=True,
    )


def track(
    *,
    label: str,
    description: str,
    source: Path,
    destination: Path,
    output_dir: Path,
    preview_duration_s: float,
    run: str,
) -> dict[str, Any]:
    export_mp3(source, destination, preview_duration_s)
    return {
        "label": label,
        "description": description,
        "path": f"audio_samples/{destination.relative_to(output_dir).as_posix()}",
        "run": run,
        "full_stream_duration_s": duration_s(source),
        "sha256": sha256(destination),
    }


def main() -> None:
    args = parse_args()
    groups = []
    for target_lang, language, target_name, run_prefix in LANGUAGES:
        for speed_factor, speed_label, speed_slug, run_speed in SPEEDS:
            group_dir = args.output_dir / f"{run_prefix}_{run_speed}_{SAMPLE_ID}"
            openai_run = f"{run_prefix}_openai_chunk960_{run_speed}"
            gemini_run = f"{run_prefix}_gemini_chunk960_{run_speed}"
            kit_run = f"{run_prefix}_kit_chunk960_{run_speed}"
            openai_dir = args.raw_root / "openai_gemini" / openai_run / SAMPLE_DIR
            gemini_dir = args.raw_root / "openai_gemini" / gemini_run / SAMPLE_DIR
            kit_dir = args.raw_root / "kit" / kit_run / SAMPLE_DIR
            tracks = [
                track(
                    label=f"Source English ({speed_label})",
                    description="Exact paced source stream sent to the systems.",
                    source=openai_dir / "source_stream_24000.wav",
                    destination=group_dir / f"source_en_{speed_slug}.mp3",
                    output_dir=args.output_dir,
                    preview_duration_s=args.preview_duration_s,
                    run=f"acl6060_live_{openai_run}",
                ),
                track(
                    label="GPT-realtime-translate target",
                    description=f"Captured {target_name} target speech; no target time-stretch.",
                    source=openai_dir / "target_audio_24000.wav",
                    destination=group_dir / f"gpt_realtime_translate_{target_lang}.mp3",
                    output_dir=args.output_dir,
                    preview_duration_s=args.preview_duration_s,
                    run=f"acl6060_live_{openai_run}",
                ),
                track(
                    label="Gemini 3.5 Live Translate target",
                    description=f"Captured {target_name} target speech; no target time-stretch.",
                    source=gemini_dir / "target_audio_24000.wav",
                    destination=group_dir / f"gemini_3p5_live_translate_{target_lang}.mp3",
                    output_dir=args.output_dir,
                    preview_duration_s=args.preview_duration_s,
                    run=f"acl6060_live_{gemini_run}",
                ),
                track(
                    label="KIT high-quality target",
                    description=(
                        f"Captured mixed/high_quality {target_name} target speech; "
                        "no target time-stretch."
                    ),
                    source=kit_dir / "target_tts.wav",
                    destination=group_dir / f"kit_high_quality_{target_lang}.mp3",
                    output_dir=args.output_dir,
                    preview_duration_s=args.preview_duration_s,
                    run=f"acl6060_live_{kit_run}",
                ),
            ]
            groups.append(
                {
                    "language": language,
                    "target_lang": target_lang,
                    "target_language_name": target_name,
                    "speed_factor": speed_factor,
                    "speed_label": speed_label,
                    "anchor": f"audio-{target_lang}-{speed_slug}",
                    "tracks": tracks,
                }
            )
    manifest = {
        "title": "ACL6060 multilingual source-speed audio samples",
        "description": (
            "The same ACL6060 talk is shown for En-Zh, En-De, and En-Ja at "
            "1x, 1.25x, and 1.5x source speed. Each group contains the exact "
            "source stream plus GPT, Gemini, and KIT target speech."
        ),
        "sample_id": SAMPLE_ID,
        "sample_index": 4,
        "chunk_ms": 960,
        "preview_start_s": 0,
        "preview_duration_s": args.preview_duration_s,
        "audio_codec": "MP3 mono 24 kHz 64 kbps",
        "groups": groups,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"groups": len(groups), "tracks": sum(len(g["tracks"]) for g in groups)}))


if __name__ == "__main__":
    main()
