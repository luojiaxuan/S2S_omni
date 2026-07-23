#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


LANGUAGES = [("zh", "En-Zh"), ("de", "En-De"), ("ja", "En-Ja")]
SPEEDS = [1.0, 1.25, 1.5]
SYSTEMS = [
    ("openai", "GPT-realtime-translate"),
    ("gemini", "Gemini 3.5-live-translate"),
    ("kit", "KIT Lecture Translator"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the ACL6060 3x3x3 full-table TSV.")
    parser.add_argument(
        "--artifact-base",
        type=Path,
        default=Path("projects/acl6060_s2s_metrics_seed/artifacts"),
    )
    parser.add_argument("--output-tsv", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--chunk-ms", type=int, default=960)
    return parser.parse_args()


def speed_tag(speed: float) -> str:
    text = ("%g" % speed).replace(".", "p")
    return f"speed{text}"


def speed_display(speed: float) -> str:
    return f"{speed:g}x"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv_scores(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            metric = row.get("metric")
            value = row.get("value")
            if not metric or value is None:
                continue
            try:
                out[metric] = float(value)
            except ValueError:
                continue
    return out


def candidate_run_dirs(artifact_base: Path, provider: str, lang: str, chunk_ms: int, speed: float) -> list[Path]:
    tag = f"{provider}_chunk{chunk_ms}_{speed_tag(speed)}"
    return [
        artifact_base / f"acl6060_live_en{lang}_{tag}",
        artifact_base / f"acl6060_live_{lang}_{tag}",
        artifact_base / f"acl6060_live_{tag}" if lang == "zh" else artifact_base / "__missing__",
    ]


def find_run_dir(artifact_base: Path, provider: str, lang: str, chunk_ms: int, speed: float) -> Path | None:
    for path in candidate_run_dirs(artifact_base, provider, lang, chunk_ms, speed):
        if (path / "instances.log").exists() and (path / "run_config.json").exists():
            return path
    return None


def provider_config(provider: str, lang: str, chunk_ms: int) -> str:
    if provider == "kit":
        return (
            f"chunk={chunk_ms}ms; format=mixed; ttsQualityMode=high_quality; "
            f"language={lang},en; target-audio ASR"
        )
    return f"chunk={chunk_ms}ms; target={lang}; live text transcript"


def load_xcomet_score(run_dir: Path) -> float | None:
    for path in [run_dir / "xcomet_xl" / "summary.json", run_dir / "xcomet_xl_summary.json"]:
        data = read_json(path)
        for key in ["xcomet_xl", "xcomet_xl_score", "system_score", "score"]:
            if data.get(key) is not None:
                return float(data[key])
    return None


def build_row(
    artifact_base: Path,
    lang: str,
    language: str,
    speed: float,
    provider: str,
    system: str,
    chunk_ms: int,
) -> dict[str, Any]:
    run_dir = find_run_dir(artifact_base, provider, lang, chunk_ms, speed)
    row: dict[str, Any] = {
        "Language": language,
        "Speedup": speed_display(speed),
        "System": system,
        "Config": provider_config(provider, lang, chunk_ms),
        "BLEU": "",
        "XCOMET-XL": "",
        "LongYAAL": "",
        "Ending Offset": "",
        "status": "missing",
        "run_dir": "",
    }
    if run_dir is None:
        return row

    row["status"] = "has_run"
    row["run_dir"] = str(run_dir)
    omnisteval = read_json(run_dir / "omnisteval_longform" / "summary.json")
    scores = omnisteval.get("scores") if isinstance(omnisteval.get("scores"), dict) else {}
    if scores:
        row["BLEU"] = f"{float(scores.get('BLEU')):.4f}" if scores.get("BLEU") is not None else ""
        row["LongYAAL"] = (
            f"{float(scores.get('LongYAAL (CU)')):.4f}"
            if scores.get("LongYAAL (CU)") is not None
            else ""
        )
        if omnisteval.get("ending_offset_ca_ms_mean") is not None:
            row["Ending Offset"] = f"{float(omnisteval['ending_offset_ca_ms_mean']):.4f}"
        row["status"] = "has_longyaal"
    else:
        rasst_scores = read_tsv_scores(run_dir / "eval_results.tsv")
        if rasst_scores.get("BLEU") is not None:
            row["BLEU"] = f"{float(rasst_scores['BLEU']):.4f}"

    xcomet = load_xcomet_score(run_dir)
    if xcomet is not None:
        row["XCOMET-XL"] = f"{xcomet:.6f}"
        if row["status"] == "has_longyaal":
            row["status"] = "complete_without_manual_check"
    return row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Language", "Speedup", "System", "Config", "BLEU", "XCOMET-XL", "LongYAAL", "Ending Offset"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = [
        build_row(args.artifact_base, lang, language, speed, provider, system, args.chunk_ms)
        for lang, language in LANGUAGES
        for speed in SPEEDS
        for provider, system in SYSTEMS
    ]
    write_tsv(args.output_tsv, rows)
    write_jsonl(args.output_jsonl, rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1
    print(
        json.dumps(
            {
                "rows": len(rows),
                "status_counts": status_counts,
                "output_tsv": str(args.output_tsv),
                "output_jsonl": str(args.output_jsonl),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
