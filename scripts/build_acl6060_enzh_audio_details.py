#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects/acl6060_s2s_metrics_seed"
ARTIFACTS = PROJECT / "artifacts"
DEFAULT_KIT_ROOT = Path("/tmp/acl6060_kit_live_sweep")
DEFAULT_SOURCE_TEXT = Path(
    "/tmp/rasst_main_result_data/main_result/inputs/acl_zh/source_text.txt"
)
SYSTEMS = {
    "openai": ("GPT realtime translate", "GPT-realtime-translate"),
    "gemini": ("Gemini 3.5 live translate", "Gemini 3.5-live-translate"),
    "kit": ("KIT Lecture Translator", "KIT Lecture Translator"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build local ACL6060 En-Zh GPT/Gemini/KIT audio details."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rerun-root", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, default=DEFAULT_KIT_ROOT)
    parser.add_argument("--source-text", type=Path, default=DEFAULT_SOURCE_TEXT)
    parser.add_argument(
        "--speeds",
        default="1p25,1p5",
        help="Comma-separated raw speed tags.",
    )
    parser.add_argument(
        "--run-ids",
        default="2022.acl-long.268,2022.acl-long.590",
        help="Comma-separated ACL talk ids.",
    )
    parser.add_argument("--clip-seconds", type=float, default=60.0)
    parser.add_argument("--text-segments", type=int, default=8)
    parser.add_argument("--audio-bitrate", default="64k")
    parser.add_argument(
        "--normalize-loudness",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def kit_artifact_dir(speed: str) -> Path:
    return ARTIFACTS / f"acl6060_live_enzh_kit_chunk960_speed{speed}"


def rerun_dir(root: Path, provider: str, speed: str) -> Path:
    return root / f"{provider}_speed{speed}"


def kit_raw_run_dir(
    artifact: Path,
    response: dict[str, Any],
    kit_root: Path,
) -> Path:
    raw_tag = artifact.name.removeprefix("acl6060_live_")
    return kit_root / raw_tag / Path(str(response["run_dir"])).name


def one_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} under {directory}, got {matches}")
    return matches[0]


def audio_duration_s(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def encode_clip(
    source: Path,
    target: Path,
    seconds: float,
    bitrate: str,
    normalize_loudness: bool,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
    ]
    if seconds > 0:
        command.extend(["-t", str(seconds)])
    if normalize_loudness:
        command.extend(["-af", "loudnorm=I=-20:TP=-2:LRA=11"])
    command.extend(
        [
            "-vn",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            str(target),
        ]
    )
    subprocess.run(command, check=True)


def grouped_segments(path: Path) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        groups[int(row["docid"])].append(row)
    return dict(groups)


def source_segments(
    source_text: Path,
    groups: dict[int, list[dict[str, Any]]],
) -> dict[int, list[str]]:
    lines = source_text.read_text(encoding="utf-8").splitlines()
    if sum(len(rows) for rows in groups.values()) != len(lines):
        raise ValueError("source text and resegmented segment counts differ")
    result: dict[int, list[str]] = {}
    cursor = 0
    for docid in sorted(groups):
        count = len(groups[docid])
        result[docid] = lines[cursor : cursor + count]
        cursor += count
    return result


def speed_label(speed: str) -> str:
    return {"1p25": "1.25x", "1p5": "1.5x"}.get(speed, speed.replace("p", "."))


def speed_value(speed: str) -> float:
    return float(speed.replace("p", "."))


def canonical_metrics() -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_jsonl(ARTIFACTS / "acl6060_full_table.jsonl"):
        if row["Language"] != "En-Zh":
            continue
        result[(str(row["Speedup"]), str(row["System"]))] = {
            "BLEU": str(row["BLEU"]),
            "XCOMET-XL": str(row["XCOMET-XL"]),
        }
    return result


def validate_rerun_config(path: Path, provider: str, speed: str) -> dict[str, Any]:
    config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    expected = {
        "provider": provider,
        "target_lang": "zh",
        "chunk_ms": 960,
        "pace": True,
        "save_output_audio": True,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if abs(float(config["speed_factor"]) - speed_value(speed)) > 1e-9:
        mismatches["speed_factor"] = (config["speed_factor"], speed_value(speed))
    if mismatches:
        raise ValueError(f"unexpected rerun config in {path}: {mismatches}")
    return config


def render_text(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def render_system(system: dict[str, Any]) -> str:
    metric = system["canonical_metric"]
    return f"""
<section class="system system-{html.escape(system["provider"])}">
  <header>
    <div>
      <p class="provider">{html.escape(system["label"])}</p>
      <p class="metric">canonical full wav · BLEU {html.escape(metric["BLEU"])} · XCOMET-XL {html.escape(metric["XCOMET-XL"])}</p>
    </div>
    <p class="duration">{system["display_duration_s"]:.1f}s audio</p>
  </header>
  <audio controls preload="metadata" src="{html.escape(system["target_clip"])}"></audio>
  <p class="provenance">{html.escape(system["provenance"])}</p>
  <details>
    <summary>{html.escape(system["transcript_label"])}</summary>
    <p>{render_text(system["transcript"])}</p>
  </details>
</section>"""


def render_page(rows: list[dict[str, Any]], clip_seconds: float) -> str:
    samples = []
    for row in rows:
        samples.append(
            f"""
<article class="sample">
  <header class="sample-header">
    <div>
      <p class="speed">{html.escape(row["speed_label"])}</p>
      <h2>{html.escape(row["run_id"])}</h2>
    </div>
    <p class="meta">960ms input chunks · first {clip_seconds:g}s sped source</p>
  </header>
  <div class="context">
    <section>
      <h3>English source audio</h3>
      <audio controls preload="metadata" src="{html.escape(row["source_clip"])}"></audio>
      <details>
        <summary>Source transcript</summary>
        <p>{render_text(row["source_excerpt"])}</p>
      </details>
    </section>
    <section>
      <h3>Human Chinese reference</h3>
      <p>{render_text(row["reference_excerpt"])}</p>
    </section>
  </div>
  <div class="systems">
    {''.join(render_system(system) for system in row["systems"])}
  </div>
</article>"""
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ACL6060 En-Zh audio details</title>
<style>
:root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #f7f8fa; color: #20242a; }}
main {{ width: min(1280px, calc(100% - 32px)); margin: 32px auto 64px; }}
h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
h2 {{ margin: 2px 0 0; font-size: 20px; letter-spacing: 0; }}
h3 {{ margin: 0 0 10px; font-size: 14px; letter-spacing: 0; }}
p {{ line-height: 1.55; }}
.intro {{ margin: 0 0 24px; max-width: 980px; color: #525b66; }}
.sample {{ margin: 18px 0; padding: 20px; background: #fff; border: 1px solid #d9dee5; border-radius: 8px; }}
.sample-header, .system header {{ display: flex; justify-content: space-between; gap: 20px; align-items: flex-start; }}
.sample-header {{ border-bottom: 1px solid #e7eaee; padding-bottom: 14px; }}
.speed {{ margin: 0; color: #006f5f; font-weight: 700; }}
.meta, .duration, .metric, .provenance {{ margin: 0; color: #66707b; font-size: 13px; }}
.meta, .duration {{ text-align: right; }}
.context {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 18px 0; }}
.context section {{ min-width: 0; }}
.context > section > p {{ max-height: 180px; overflow: auto; margin: 0; padding-right: 8px; font-size: 14px; }}
.systems {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); border-top: 1px solid #e7eaee; }}
.system {{ min-width: 0; padding: 18px; border-right: 1px solid #e7eaee; }}
.system:first-child {{ padding-left: 0; }}
.system:last-child {{ padding-right: 0; border-right: 0; }}
.provider {{ margin: 0 0 4px; font-weight: 750; }}
.system-openai .provider {{ color: #126344; }}
.system-gemini .provider {{ color: #3158a6; }}
.system-kit .provider {{ color: #9a4c00; }}
audio {{ display: block; width: 100%; height: 40px; margin: 12px 0; }}
details {{ margin-top: 12px; }}
summary {{ cursor: pointer; color: #333b44; font-size: 14px; font-weight: 650; }}
details p {{ max-height: 250px; overflow: auto; margin: 10px 0 0; padding-right: 8px; font-size: 14px; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
@media (max-width: 900px) {{
  .systems, .context {{ grid-template-columns: 1fr; }}
  .system, .system:first-child, .system:last-child {{ padding: 18px 0; border-right: 0; border-bottom: 1px solid #e7eaee; }}
  .system:last-child {{ border-bottom: 0; }}
}}
@media (max-width: 620px) {{
  main {{ width: min(100% - 20px, 1280px); margin-top: 20px; }}
  .sample-header, .system header {{ display: grid; grid-template-columns: 1fr; }}
  .meta, .duration {{ text-align: left; }}
}}
</style>
</head>
<body>
<main>
  <h1>ACL6060 En-Zh audio details</h1>
  <p class="intro">Two stable ACL talks at 1.25x and 1.5x source speed. GPT and Gemini audio comes from new first-60s qualitative reruns using the canonical model, target language, 960ms chunk, and speed settings. KIT audio is the first 60s of the exact persisted canonical full-wav <code>high_quality</code> target. Because KIT is a time-based target excerpt from a full-talk run, it is not source-window aligned with the GPT/Gemini reruns and may cover less source content. Use this page for speech-quality inspection, not a source-aligned ranking. Presentation MP3s are loudness-normalized; raw WAVs remain unchanged.</p>
  {''.join(samples)}
</main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    speeds = [value.strip() for value in args.speeds.split(",") if value.strip()]
    run_ids = [value.strip() for value in args.run_ids.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = canonical_metrics()
    rows: list[dict[str, Any]] = []

    for speed in speeds:
        kit_artifact = kit_artifact_dir(speed)
        kit_config = json.loads(
            (kit_artifact / "run_config.json").read_text(encoding="utf-8")
        )
        if kit_config["kit_tts_quality_mode"] != "high_quality":
            raise ValueError(f"{kit_artifact} is not high_quality")
        if kit_config["target_lang"] != "zh" or kit_config["chunk_ms"] != 960:
            raise ValueError(f"unexpected KIT config in {kit_artifact}")

        kit_responses = read_jsonl(kit_artifact / "responses.jsonl")
        kit_response_by_id = {str(row["run_id"]): row for row in kit_responses}
        groups = grouped_segments(
            kit_artifact / "omnisteval_longform/instances.resegmented.jsonl"
        )
        sources = source_segments(args.source_text, groups)

        reruns: dict[str, dict[str, Any]] = {}
        for provider in ["openai", "gemini"]:
            provider_dir = rerun_dir(args.rerun_root, provider, speed)
            config = validate_rerun_config(provider_dir, provider, speed)
            responses = read_jsonl(provider_dir / "responses.jsonl")
            reruns[provider] = {
                "dir": provider_dir,
                "config": config,
                "responses": {str(row["run_id"]): row for row in responses},
            }

        for run_id in run_ids:
            kit_response = kit_response_by_id[run_id]
            docid = int(kit_response["index"])
            safe_id = run_id.replace(".", "_")
            row_dir = args.output_dir / f"speed_{speed}" / safe_id
            source_run_dir = (
                reruns["openai"]["dir"] / f"{docid:03d}_{run_id}"
            )
            source_wav = one_file(source_run_dir, "source_stream_*.wav")
            source_clip = row_dir / "source_first60s.mp3"
            encode_clip(
                source_wav,
                source_clip,
                args.clip_seconds,
                args.audio_bitrate,
                args.normalize_loudness,
            )

            systems: list[dict[str, Any]] = []
            for provider in ["openai", "gemini"]:
                response = reruns[provider]["responses"][run_id]
                run_dir = reruns[provider]["dir"] / f"{docid:03d}_{run_id}"
                target_wav = one_file(run_dir, "target_audio_*.wav")
                target_clip = row_dir / f"{provider}_target_full.mp3"
                encode_clip(
                    target_wav,
                    target_clip,
                    0.0,
                    args.audio_bitrate,
                    args.normalize_loudness,
                )
                label, canonical_name = SYSTEMS[provider]
                systems.append(
                    {
                        "provider": provider,
                        "label": label,
                        "target_clip": target_clip.relative_to(
                            args.output_dir
                        ).as_posix(),
                        "raw_target_wav": str(target_wav),
                        "raw_target_duration_s": audio_duration_s(target_wav),
                        "display_duration_s": audio_duration_s(target_wav),
                        "transcript": str(response["prediction"]),
                        "transcript_label": "Target transcript",
                        "error_count": int(response["error_count"]),
                        "provenance": (
                            "Same-config first-60s qualitative rerun; not the "
                            "canonical full-wav session."
                        ),
                        "canonical_metric": metrics[
                            (speed_label(speed), canonical_name)
                        ],
                        "run_config": reruns[provider]["config"],
                    }
                )

            kit_raw_dir = kit_raw_run_dir(
                kit_artifact,
                kit_response,
                args.kit_root,
            )
            kit_target_wav = kit_raw_dir / "target_tts.wav"
            if not kit_target_wav.exists():
                raise FileNotFoundError(kit_target_wav)
            kit_target_clip = row_dir / "kit_target_first60s.mp3"
            encode_clip(
                kit_target_wav,
                kit_target_clip,
                args.clip_seconds,
                args.audio_bitrate,
                args.normalize_loudness,
            )
            segment_rows = groups[docid][: args.text_segments]
            kit_label, kit_canonical_name = SYSTEMS["kit"]
            systems.append(
                {
                    "provider": "kit",
                    "label": kit_label,
                    "target_clip": kit_target_clip.relative_to(
                        args.output_dir
                    ).as_posix(),
                    "raw_target_wav": str(kit_target_wav),
                    "raw_target_duration_s": audio_duration_s(kit_target_wav),
                    "display_duration_s": min(
                        args.clip_seconds,
                        audio_duration_s(kit_target_wav),
                    ),
                    "transcript": " ".join(
                        str(row["prediction"]) for row in segment_rows
                    ),
                    "transcript_label": (
                        "First reference-aligned full-talk segments "
                        "(not aligned to this audio window)"
                    ),
                    "error_count": 0,
                    "provenance": (
                        "First 60s of exact persisted canonical full-wav "
                        "high_quality target_tts.wav. This target-time excerpt "
                        "is not source-window aligned with the reruns."
                    ),
                    "canonical_metric": metrics[
                        (speed_label(speed), kit_canonical_name)
                    ],
                    "run_config": kit_config,
                }
            )

            rows.append(
                {
                    "speed": speed,
                    "speed_label": speed_label(speed),
                    "run_id": run_id,
                    "docid": docid,
                    "source_clip": source_clip.relative_to(
                        args.output_dir
                    ).as_posix(),
                    "raw_source_wav": str(source_wav),
                    "source_duration_s": audio_duration_s(source_wav),
                    "source_excerpt": " ".join(
                        sources[docid][: args.text_segments]
                    ),
                    "reference_excerpt": " ".join(
                        str(row["reference"]) for row in segment_rows
                    ),
                    "systems": systems,
                }
            )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "ACL6060 En-Zh GPT/Gemini/KIT qualitative audio details",
        "clip_seconds": args.clip_seconds,
        "selection": run_ids,
        "selection_note": (
            "Stable talks 268 and 590; talk 367 excluded because it is a known "
            "KIT partial-output failure."
        ),
        "provenance_note": (
            "GPT/Gemini are same-config first-60s reruns because historical "
            "canonical raw logs redacted audio payloads. KIT is an excerpt from "
            "the exact canonical full-wav target audio."
        ),
        "display_audio_processing": {
            "format": "MP3",
            "sample_rate": 24000,
            "channels": 1,
            "bitrate": args.audio_bitrate,
            "loudness_normalized": args.normalize_loudness,
            "loudnorm": "I=-20:TP=-2:LRA=11" if args.normalize_loudness else "",
        },
        "upload_status": "PENDING_HF_UPLOAD",
        "rows": rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "index.html").write_text(
        render_page(rows, args.clip_seconds),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "samples": len(rows),
                "systems": sum(len(row["systems"]) for row in rows),
                "audio_files": len(rows) * 4,
            }
        )
    )


if __name__ == "__main__":
    main()
