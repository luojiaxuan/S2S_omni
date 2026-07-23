#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Unbabel/XCOMET-XL over ACL6060 segments.")
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--model-name", default="Unbabel/XCOMET-XL")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--reference-free", action="store_true")
    parser.add_argument("--write-run-summaries", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=False) for row in rows) + "\n",
        encoding="utf-8",
    )


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


def comet_data(rows: list[dict[str, Any]], reference_free: bool) -> list[dict[str, str]]:
    data = []
    for row in rows:
        item = {"src": str(row["source"]), "mt": str(row["hypothesis"])}
        if not reference_free:
            item["ref"] = str(row["reference"])
        data.append(item)
    return data


def write_run_summaries(rows: list[dict[str, Any]], model_name: str, mode: str) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        run_dir = str(row.get("run_dir") or "")
        if not run_dir:
            continue
        grouped.setdefault(run_dir, []).append(row)
    for run_dir, run_rows in grouped.items():
        out_dir = Path(run_dir) / "xcomet_xl"
        write_jsonl(out_dir / "segments.jsonl", run_rows)
        summary = {
            "xcomet_xl": weighted_mean(run_rows, "xcomet_xl_score"),
            "xcomet_xl_model": model_name,
            "xcomet_xl_mode": mode,
            "segments": len(run_rows),
            "output_jsonl": str(out_dir / "segments.jsonl"),
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_jsonl)
    from comet import download_model, load_from_checkpoint

    model_path = download_model(args.model_name)
    model = load_from_checkpoint(model_path)
    output = model.predict(comet_data(rows, args.reference_free), batch_size=args.batch_size, gpus=args.gpus)
    scores = [float(score) for score in output.scores]
    if len(scores) != len(rows):
        raise RuntimeError(f"expected {len(rows)} scores, got {len(scores)}")
    out_rows = []
    mode = "source_hypothesis" if args.reference_free else "source_hypothesis_reference"
    for row, score in zip(rows, scores):
        out = dict(row)
        out["xcomet_xl_score"] = score
        out["xcomet_xl_model"] = args.model_name
        out["xcomet_xl_mode"] = mode
        out_rows.append(out)
    write_jsonl(args.output_jsonl, out_rows)
    if args.write_run_summaries:
        write_run_summaries(out_rows, args.model_name, mode)
    system_score = weighted_mean(out_rows, "xcomet_xl_score")
    summary = {
        "xcomet_xl": system_score,
        "xcomet_xl_model": args.model_name,
        "xcomet_xl_mode": mode,
        "segments": len(out_rows),
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
