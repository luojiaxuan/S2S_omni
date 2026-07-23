#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_acl6060_full_table import LANGUAGES, SPEEDS, SYSTEMS, find_run_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LongYAAL/BLEU and XCOMET-XL prep/scoring for ACL6060 full-table artifacts."
    )
    parser.add_argument(
        "--artifact-base",
        type=Path,
        default=ROOT / "projects/acl6060_s2s_metrics_seed/artifacts",
    )
    parser.add_argument("--chunk-ms", type=int, default=960)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--omnisteval-bin", default="")
    parser.add_argument("--run-omnisteval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--build-xcomet-input", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-xcomet", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--xcomet-model-name", default="Unbabel/XCOMET-XL")
    parser.add_argument("--xcomet-batch-size", type=int, default=4)
    parser.add_argument("--xcomet-gpus", type=int, default=1)
    parser.add_argument("--reference-free-xcomet", action="store_true")
    parser.add_argument(
        "--combined-xcomet-input",
        type=Path,
        default=None,
        help="Default: <artifact-base>/acl6060_xcomet_xl/input_all.jsonl",
    )
    parser.add_argument(
        "--combined-xcomet-output",
        type=Path,
        default=None,
        help="Default: <artifact-base>/acl6060_xcomet_xl/scores_all.jsonl",
    )
    parser.add_argument(
        "--combined-xcomet-summary",
        type=Path,
        default=None,
        help="Default: <artifact-base>/acl6060_xcomet_xl/summary_all.json",
    )
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=ROOT / "projects/acl6060_s2s_metrics_seed/artifacts/acl6060_full_table.tsv",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=ROOT / "projects/acl6060_s2s_metrics_seed/artifacts/acl6060_full_table.jsonl",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print(json.dumps({"cmd": cmd}, ensure_ascii=False), flush=True)
    subprocess.run(cmd, check=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def expected_run_dirs(artifact_base: Path, chunk_ms: int) -> list[tuple[str, float, str, Path]]:
    out: list[tuple[str, float, str, Path]] = []
    seen: set[Path] = set()
    for lang, _language in LANGUAGES:
        for speed in SPEEDS:
            for provider, _system in SYSTEMS:
                run_dir = find_run_dir(artifact_base, provider, lang, chunk_ms, speed)
                if run_dir is None or run_dir in seen:
                    continue
                seen.add(run_dir)
                out.append((lang, speed, provider, run_dir))
    return out


def run_omnisteval(args: argparse.Namespace, run_dir: Path) -> None:
    summary = run_dir / "omnisteval_longform" / "summary.json"
    scores = run_dir / "omnisteval_longform" / "scores.tsv"
    if summary.exists() and scores.exists():
        return
    cmd = [
        args.python_bin,
        str(SCRIPT_DIR / "run_acl6060_omnisteval.py"),
        "--run-dir",
        str(run_dir),
    ]
    if args.omnisteval_bin:
        cmd.extend(["--omnisteval-bin", args.omnisteval_bin])
    run_cmd(cmd)


def build_xcomet_input(args: argparse.Namespace, run_dir: Path) -> Path | None:
    resegmented = run_dir / "omnisteval_longform" / "instances.resegmented.jsonl"
    if not resegmented.exists():
        return None
    output = run_dir / "xcomet_xl" / "input.jsonl"
    if not output.exists():
        run_cmd(
            [
                args.python_bin,
                str(SCRIPT_DIR / "build_acl6060_xcomet_input.py"),
                "--run-dir",
                str(run_dir),
                "--output-jsonl",
                str(output),
            ]
        )
    return output


def xcomet_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    base = args.artifact_base / "acl6060_xcomet_xl"
    return (
        args.combined_xcomet_input or (base / "input_all.jsonl"),
        args.combined_xcomet_output or (base / "scores_all.jsonl"),
        args.combined_xcomet_summary or (base / "summary_all.json"),
    )


def build_combined_xcomet_input(input_paths: list[Path], output_path: Path) -> int:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in input_paths:
        for row in read_jsonl(path):
            row_id = str(row.get("xcomet_id") or "")
            if row_id and row_id in seen_ids:
                continue
            if row_id:
                seen_ids.add(row_id)
            rows.append(row)
    write_jsonl(output_path, rows)
    return len(rows)


def weighted_mean(rows: list[dict[str, Any]], score_key: str) -> float | None:
    values = []
    weight_sum = 0.0
    for row in rows:
        if row.get(score_key) is None:
            continue
        weight = float(row.get("weight_chars") or 1.0)
        values.append(float(row[score_key]) * weight)
        weight_sum += weight
    if not values:
        return None
    return sum(values) / weight_sum if weight_sum > 0 else sum(values) / len(values)


def split_xcomet_scores(scored_jsonl: Path) -> int:
    rows = read_jsonl(scored_jsonl)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        run_dir = str(row.get("run_dir") or "")
        if run_dir:
            grouped.setdefault(run_dir, []).append(row)
    for run_dir, run_rows in grouped.items():
        out_dir = Path(run_dir) / "xcomet_xl"
        write_jsonl(out_dir / "segments.jsonl", run_rows)
        first = run_rows[0]
        summary = {
            "xcomet_xl": weighted_mean(run_rows, "xcomet_xl_score"),
            "xcomet_xl_model": first.get("xcomet_xl_model"),
            "xcomet_xl_mode": first.get("xcomet_xl_mode"),
            "segments": len(run_rows),
            "output_jsonl": str(out_dir / "segments.jsonl"),
            "combined_output_jsonl": str(scored_jsonl),
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return len(grouped)


def run_xcomet(args: argparse.Namespace, input_path: Path, output_path: Path, summary_path: Path) -> None:
    if output_path.exists() and summary_path.exists():
        return
    cmd = [
        args.python_bin,
        str(SCRIPT_DIR / "run_acl6060_xcomet_xl.py"),
        "--input-jsonl",
        str(input_path),
        "--output-jsonl",
        str(output_path),
        "--summary-json",
        str(summary_path),
        "--model-name",
        args.xcomet_model_name,
        "--batch-size",
        str(args.xcomet_batch_size),
        "--gpus",
        str(args.xcomet_gpus),
        "--write-run-summaries",
    ]
    if args.reference_free_xcomet:
        cmd.append("--reference-free")
    run_cmd(cmd)


def build_table(args: argparse.Namespace) -> None:
    run_cmd(
        [
            args.python_bin,
            str(SCRIPT_DIR / "build_acl6060_full_table.py"),
            "--artifact-base",
            str(args.artifact_base),
            "--output-tsv",
            str(args.output_tsv),
            "--output-jsonl",
            str(args.output_jsonl),
            "--chunk-ms",
            str(args.chunk_ms),
        ]
    )


def main() -> None:
    args = parse_args()
    run_dirs = expected_run_dirs(args.artifact_base, args.chunk_ms)
    xcomet_inputs: list[Path] = []
    for _lang, _speed, _provider, run_dir in run_dirs:
        if args.run_omnisteval:
            run_omnisteval(args, run_dir)
        if args.build_xcomet_input:
            input_path = build_xcomet_input(args, run_dir)
            if input_path is not None:
                xcomet_inputs.append(input_path)

    combined_input, combined_output, combined_summary = xcomet_paths(args)
    combined_rows = build_combined_xcomet_input(xcomet_inputs, combined_input) if xcomet_inputs else 0
    if args.run_xcomet and combined_rows:
        run_xcomet(args, combined_input, combined_output, combined_summary)
    xcomet_run_summaries = split_xcomet_scores(combined_output) if combined_output.exists() else 0
    build_table(args)
    print(
        json.dumps(
            {
                "run_dirs": len(run_dirs),
                "xcomet_input_files": len(xcomet_inputs),
                "combined_xcomet_rows": combined_rows,
                "xcomet_run_summaries": xcomet_run_summaries,
                "combined_xcomet_input": str(combined_input),
                "output_tsv": str(args.output_tsv),
                "output_jsonl": str(args.output_jsonl),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
