#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_BASE = ROOT / "projects/acl6060_s2s_metrics_seed/artifacts"
DEFAULT_HIGH = ARTIFACT_BASE / "acl6060_live_enzh_kit_chunk960_speed1"
DEFAULT_LOW = ARTIFACT_BASE / "acl6060_live_enzh_kit_low_latency_chunk960_speed1"
DEFAULT_PREFIX = ARTIFACT_BASE / "acl6060_kit_enzh_speed1_quality_mode_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the ACL6060 KIT high-quality versus low-latency one-cell comparison."
    )
    parser.add_argument("--high-quality-run-dir", type=Path, default=DEFAULT_HIGH)
    parser.add_argument("--low-latency-run-dir", type=Path, default=DEFAULT_LOW)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-tsv", type=Path, default=DEFAULT_PREFIX.with_suffix(".tsv"))
    parser.add_argument("--output-json", type=Path, default=DEFAULT_PREFIX.with_suffix(".json"))
    parser.add_argument(
        "--canonical-table-jsonl",
        type=Path,
        default=ARTIFACT_BASE / "acl6060_full_table.jsonl",
    )
    parser.add_argument(
        "--per-talk-tsv",
        type=Path,
        default=DEFAULT_PREFIX.with_name(DEFAULT_PREFIX.name + "_per_talk").with_suffix(".tsv"),
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--omnisteval-bin", default="")
    parser.add_argument(
        "--raw-local-staging",
        default=(
            "/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/"
            "acl6060_kit_live_sweep/enzh_kit_chunk960_speed1_low_latency"
        ),
    )
    parser.add_argument("--raw-audio-upload-status", default="PENDING_HF_UPLOAD")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_config_path(value: Any, dataset_root: Path | None) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    if dataset_root is not None and "main_result/" in path.as_posix():
        rel = path.as_posix().split("main_result/", 1)[1]
        candidate = dataset_root / "main_result" / rel
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


def config_diff(high: dict[str, Any], low: dict[str, Any]) -> dict[str, list[Any]]:
    keys = sorted(set(high) | set(low))
    return {key: [high.get(key), low.get(key)] for key in keys if high.get(key) != low.get(key)}


def aggregate_metrics(run_dir: Path) -> dict[str, float]:
    omni = read_json(run_dir / "omnisteval_longform/summary.json")
    xcomet = read_json(run_dir / "xcomet_xl/summary.json")
    responses = read_jsonl(run_dir / "responses.jsonl")
    return {
        "BLEU": float(omni["scores"]["BLEU"]),
        "XCOMET-XL": float(xcomet["xcomet_xl"]),
        "LongYAAL": float(omni["scores"]["LongYAAL (CU)"]),
        "Ending Offset": float(omni["ending_offset_ca_ms_mean"]),
        "mean_tts_audio_chunks": statistics.mean(
            float(row["tts_audio_chunks"]) for row in responses
        ),
        "mean_target_audio_duration_ms": statistics.mean(
            float(row["target_audio_duration_ms"]) for row in responses
        ),
        "mean_prediction_units": statistics.mean(
            float(row["prediction_units"]) for row in responses
        ),
    }


def canonical_baseline_row(path: Path) -> dict[str, Any]:
    matches = [
        row
        for row in read_jsonl(path)
        if row.get("Language") == "En-Zh"
        and row.get("Speedup") == "1x"
        and row.get("System") == "KIT Lecture Translator"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one canonical En-Zh/1x/KIT row, got {len(matches)}")
    return matches[0]


def validate_canonical_baseline(row: dict[str, Any], metrics: dict[str, float]) -> None:
    fields = ["BLEU", "XCOMET-XL", "LongYAAL", "Ending Offset"]
    mismatches = {}
    for field in fields:
        canonical_text = str(row[field])
        decimals = len(canonical_text.partition(".")[2])
        tolerance = Decimal(5).scaleb(-(decimals + 1))
        difference = abs(Decimal(canonical_text) - Decimal(str(metrics[field])))
        if difference > tolerance:
            mismatches[field] = [float(canonical_text), metrics[field]]
    if mismatches:
        raise ValueError(f"canonical baseline mismatch: {mismatches}")


def consecutive_audio_groups(audio_rows: list[dict[str, Any]]) -> list[tuple[str, list[int]]]:
    groups: list[tuple[str, list[int]]] = []
    for index, row in enumerate(audio_rows):
        wav = str(row["wav"])
        if not groups or groups[-1][0] != wav:
            groups.append((wav, []))
        groups[-1][1].append(index)
    return groups


def run_per_talk(
    *,
    mode: str,
    run_dir: Path,
    audio_rows: list[dict[str, Any]],
    reference_lines: list[str],
    groups: list[tuple[str, list[int]]],
    python_bin: str,
    omnisteval_bin: str,
    work_root: Path,
) -> list[dict[str, Any]]:
    instances = read_jsonl(run_dir / "instances.log")
    if len(instances) != len(groups):
        raise ValueError(f"{mode}: {len(instances)} instances for {len(groups)} talks")
    rows = []
    for talk_index, ((wav, segment_ids), instance) in enumerate(zip(groups, instances)):
        talk = Path(wav).stem
        work_dir = work_root / mode / f"{talk_index:03d}_{talk}"
        work_dir.mkdir(parents=True, exist_ok=True)
        one_instance = dict(instance)
        one_instance["index"] = 0
        (work_dir / "instances.log").write_text(
            json.dumps(one_instance, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (work_dir / "audio.yaml").write_text(
            yaml.safe_dump(
                [audio_rows[index] for index in segment_ids],
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (work_dir / "ref.txt").write_text(
            "\n".join(reference_lines[index] for index in segment_ids) + "\n",
            encoding="utf-8",
        )
        write_json(
            work_dir / "run_config.json",
            {
                "provider": "kit",
                "target_lang": "zh",
                "speed_factor": 1.0,
                "audio_yaml": str((work_dir / "audio.yaml").resolve()),
                "ref_file": str((work_dir / "ref.txt").resolve()),
            },
        )
        command = [
            python_bin,
            str(ROOT / "scripts/run_acl6060_omnisteval.py"),
            "--run-dir",
            str(work_dir),
            "--no-resume",
        ]
        if omnisteval_bin:
            command.extend(["--omnisteval-bin", omnisteval_bin])
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        summary = read_json(work_dir / "omnisteval_longform/summary.json")
        rows.append(
            {
                "mode": mode,
                "talk": talk,
                "segments": int(summary["rows"]),
                "BLEU": float(summary["scores"]["BLEU"]),
                "LongYAAL": float(summary["scores"]["LongYAAL (CU)"]),
                "Ending Offset": float(summary["ending_offset_ca_ms_mean"]),
                "longyaal_scope": "isolated_talk_rescore_non_additive",
                "ending_offset_scope": (
                    "non_informative_partial_output_failure"
                    if talk == "2022.acl-long.367"
                    else "per_talk_component_of_aggregate_mean"
                ),
            }
        )
    return rows


def write_aggregate_tsv(
    path: Path,
    high: dict[str, float],
    low: dict[str, float],
) -> None:
    target_audio_delta = (
        low["mean_target_audio_duration_ms"] - high["mean_target_audio_duration_ms"]
    )
    fields = [
        "Mode",
        "Config",
        "BLEU",
        "XCOMET-XL",
        "LongYAAL",
        "Ending Offset",
        "Mean TTS Chunks",
        "Mean Target Audio (ms)",
        "Mean Prediction Units",
    ]
    rows = []
    for mode, values in [("high_quality", high), ("low_latency", low)]:
        rows.append(
            {
                "Mode": mode,
                "Config": (
                    "chunk=960ms; speed=1; format=mixed; "
                    f"ttsQualityMode={mode}; language=zh,en; target-audio ASR"
                ),
                "BLEU": f"{values['BLEU']:.4f}",
                "XCOMET-XL": f"{values['XCOMET-XL']:.6f}",
                "LongYAAL": f"{values['LongYAAL']:.4f}",
                "Ending Offset": f"{values['Ending Offset']:.4f}",
                "Mean TTS Chunks": f"{values['mean_tts_audio_chunks']:.1f}",
                "Mean Target Audio (ms)": f"{values['mean_target_audio_duration_ms']:.1f}",
                "Mean Prediction Units": f"{values['mean_prediction_units']:.1f}",
            }
        )
    rows.append(
        {
            "Mode": "delta (low-high)",
            "Config": "",
            "BLEU": f"{low['BLEU'] - high['BLEU']:.4f}",
            "XCOMET-XL": f"{low['XCOMET-XL'] - high['XCOMET-XL']:.6f}",
            "LongYAAL": f"{low['LongYAAL'] - high['LongYAAL']:.4f}",
            "Ending Offset": f"{low['Ending Offset'] - high['Ending Offset']:.4f}",
            "Mean TTS Chunks": (
                f"{low['mean_tts_audio_chunks'] - high['mean_tts_audio_chunks']:.1f}"
            ),
            "Mean Target Audio (ms)": f"{target_audio_delta:.1f}",
            "Mean Prediction Units": (
                f"{low['mean_prediction_units'] - high['mean_prediction_units']:.1f}"
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_per_talk_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    by_key = {(row["mode"], row["talk"]): row for row in rows}
    talks = [row["talk"] for row in rows if row["mode"] == "high_quality"]
    fields = [
        "Talk",
        "Segments",
        "High BLEU",
        "Low BLEU",
        "High LongYAAL (isolated non-additive)",
        "Low LongYAAL (isolated non-additive)",
        "LongYAAL Delta (isolated non-additive)",
        "High Ending Offset",
        "Low Ending Offset",
        "Ending Offset Delta",
        "Ending Offset Interpretation",
    ]
    output = []
    for talk in talks:
        high = by_key["high_quality", talk]
        low = by_key["low_latency", talk]
        output.append(
            {
                "Talk": talk,
                "Segments": high["segments"],
                "High BLEU": f"{high['BLEU']:.4f}",
                "Low BLEU": f"{low['BLEU']:.4f}",
                "High LongYAAL (isolated non-additive)": f"{high['LongYAAL']:.4f}",
                "Low LongYAAL (isolated non-additive)": f"{low['LongYAAL']:.4f}",
                "LongYAAL Delta (isolated non-additive)": (
                    f"{low['LongYAAL'] - high['LongYAAL']:.4f}"
                ),
                "High Ending Offset": f"{high['Ending Offset']:.4f}",
                "Low Ending Offset": f"{low['Ending Offset']:.4f}",
                "Ending Offset Delta": (
                    f"{low['Ending Offset'] - high['Ending Offset']:.4f}"
                ),
                "Ending Offset Interpretation": low["ending_offset_scope"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output)


def validate_ending_offset_reconciliation(
    rows: list[dict[str, Any]],
    high: dict[str, float],
    low: dict[str, float],
) -> dict[str, float]:
    output = {}
    for mode, aggregate in [("high_quality", high), ("low_latency", low)]:
        values = [float(row["Ending Offset"]) for row in rows if row["mode"] == mode]
        per_talk_mean = statistics.mean(values)
        if abs(per_talk_mean - aggregate["Ending Offset"]) > 1e-6:
            raise ValueError(
                f"{mode} Ending Offset mismatch: {per_talk_mean} vs "
                f"{aggregate['Ending Offset']}"
            )
        output[f"{mode}_per_talk_mean"] = per_talk_mean
        output[f"{mode}_aggregate"] = aggregate["Ending Offset"]
    return output


def main() -> None:
    args = parse_args()
    high_config = read_json(args.high_quality_run_dir / "run_config.json")
    low_config = read_json(args.low_latency_run_dir / "run_config.json")
    differences = config_diff(high_config, low_config)
    if set(differences) != {"kit_tts_quality_mode"}:
        raise ValueError(f"expected only kit_tts_quality_mode to differ, got {differences}")

    audio_path = resolve_config_path(high_config["audio_yaml"], args.dataset_root)
    ref_path = resolve_config_path(high_config["ref_file"], args.dataset_root)
    audio_rows = yaml.safe_load(audio_path.read_text(encoding="utf-8"))
    reference_lines = ref_path.read_text(encoding="utf-8").splitlines()
    if len(audio_rows) != len(reference_lines):
        raise ValueError(f"audio/reference mismatch: {len(audio_rows)} vs {len(reference_lines)}")
    groups = consecutive_audio_groups(audio_rows)

    high = aggregate_metrics(args.high_quality_run_dir)
    low = aggregate_metrics(args.low_latency_run_dir)
    canonical_row = canonical_baseline_row(args.canonical_table_jsonl)
    validate_canonical_baseline(canonical_row, high)
    with tempfile.TemporaryDirectory(prefix="acl6060_kit_quality_mode_") as tmp:
        work_root = Path(tmp)
        per_talk = run_per_talk(
            mode="high_quality",
            run_dir=args.high_quality_run_dir,
            audio_rows=audio_rows,
            reference_lines=reference_lines,
            groups=groups,
            python_bin=args.python_bin,
            omnisteval_bin=args.omnisteval_bin,
            work_root=work_root,
        )
        per_talk.extend(
            run_per_talk(
                mode="low_latency",
                run_dir=args.low_latency_run_dir,
                audio_rows=audio_rows,
                reference_lines=reference_lines,
                groups=groups,
                python_bin=args.python_bin,
                omnisteval_bin=args.omnisteval_bin,
                work_root=work_root,
            )
        )

    write_aggregate_tsv(args.output_tsv, high, low)
    write_per_talk_tsv(args.per_talk_tsv, per_talk)
    ending_offset_reconciliation = validate_ending_offset_reconciliation(
        per_talk,
        high,
        low,
    )
    delta = {key: low[key] - high[key] for key in high}
    payload = {
        "baseline_artifact": str(args.high_quality_run_dir),
        "candidate_artifact": str(args.low_latency_run_dir),
        "canonical_baseline_row": canonical_row,
        "run_config_diff": differences,
        "protocol": {
            "language": "En-Zh",
            "speed_factor": float(low_config["speed_factor"]),
            "chunk_ms": int(low_config["chunk_ms"]),
            "format": low_config["kit_format"],
            "languages": low_config["kit_languages"],
            "mt_language": low_config["kit_mt_language"],
            "audio_language": low_config["kit_audio_language"],
            "smart_chaptering": low_config["kit_smart_chaptering"],
            "availability": low_config["kit_availability"],
            "hypothesis_source": low_config["candidate_text_source"],
            "samples": len(read_jsonl(args.low_latency_run_dir / "instances.log")),
        },
        "high_quality": high,
        "low_latency": low,
        "delta_low_minus_high": delta,
        "per_talk": per_talk,
        "ending_offset_reconciliation": ending_offset_reconciliation,
        "interpretation_caveats": [
            (
                "2022.acl-long.367 is a severe partial-output failure in both modes; "
                "keep it in the formal five-talk aggregate."
            ),
            (
                "Target-unit times assume units are uniform over target audio and snap "
                "them to TTS chunk arrivals, so different chunk counts confound latency."
            ),
            (
                "Delay timestamps are clamped to source_length, so late tail arrivals "
                "are understated in both modes."
            ),
            (
                "Per-talk LongYAAL values are isolated re-scores and do not decompose "
                "the joint 468-segment aggregate; use them only as non-additive diagnostics."
            ),
            (
                "BLEU and XCOMET-XL move in opposite directions over five correlated talks; "
                "no significance test was run."
            ),
        ],
        "tracked_audit": {
            "instances": len(read_jsonl(args.low_latency_run_dir / "instances.log")),
            "responses": len(read_jsonl(args.low_latency_run_dir / "responses.jsonl")),
            "xcomet_segments": len(
                read_jsonl(args.low_latency_run_dir / "xcomet_xl/segments.jsonl")
            ),
            "config_diff_keys": sorted(differences),
            "status": "STRUCTURE_PASS_WITH_INTERPRETATION_CAVEATS",
        },
        "raw_local_staging": args.raw_local_staging,
        "raw_audio_upload_status": args.raw_audio_upload_status,
    }
    write_json(args.output_json, payload)
    print(
        json.dumps(
            {
                "output_tsv": str(args.output_tsv),
                "output_json": str(args.output_json),
                "per_talk_tsv": str(args.per_talk_tsv),
                "config_diff": differences,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
