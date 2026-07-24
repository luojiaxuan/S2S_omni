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
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def arithmetic_mean(rows: list[dict[str, Any]], score_key: str) -> float | None:
    values = [float(row[score_key]) for row in rows if row.get(score_key) is not None]
    return sum(values) / len(values) if values else None


def configure_comet_prediction_load_compatibility() -> None:
    """Keep COMET 2.2.7 compatible with PyTorch's newer pickle default."""
    import torch

    original_load = torch.load

    def load_with_comet_legacy_default(*args: Any, **kwargs: Any) -> Any:
        # note (luojiaxuan): COMET writes and immediately reads its own trusted
        # distributed prediction objects, whose custom Prediction class PyTorch
        # 2.6+ rejects when torch.load defaults to weights_only=True.
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_with_comet_legacy_default


def comet_data(rows: list[dict[str, Any]], reference_free: bool) -> list[dict[str, str]]:
    data = []
    for row in rows:
        item = {"src": str(row["source"]), "mt": str(row["hypothesis"])}
        if not reference_free:
            item["ref"] = str(row["reference"])
        data.append(item)
    return data


def attach_scores(
    rows: list[dict[str, Any]],
    model_scores: list[float],
    model_name: str,
    mode: str,
) -> list[dict[str, Any]]:
    scoreable_indices = [
        index for index, row in enumerate(rows) if row.get("fixed_xcomet_xl_score") is None
    ]
    if len(model_scores) != len(scoreable_indices):
        raise ValueError(f"expected {len(scoreable_indices)} model scores, got {len(model_scores)}")
    model_score_by_index = dict(zip(scoreable_indices, model_scores))
    out_rows = []
    for index, row in enumerate(rows):
        fixed_score = row.get("fixed_xcomet_xl_score")
        score = (
            float(fixed_score) if fixed_score is not None else float(model_score_by_index[index])
        )
        out = dict(row)
        out["xcomet_xl_score"] = score
        out["xcomet_xl_model"] = model_name
        out["xcomet_xl_mode"] = mode
        out["xcomet_xl_score_source"] = (
            "fixed_null_alignment_penalty" if fixed_score is not None else "model"
        )
        out_rows.append(out)
    return out_rows


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
        null_rows = [row for row in run_rows if row.get("null_alignment_type")]
        summary = {
            "xcomet_xl": arithmetic_mean(run_rows, "xcomet_xl_score"),
            "xcomet_xl_model": model_name,
            "xcomet_xl_mode": mode,
            "alignment_backend": "SEGALE",
            "segments": len(run_rows),
            "valid_segments": len(run_rows) - len(null_rows),
            "null_alignments": len(null_rows),
            "over_translation_alignments": sum(
                row.get("null_alignment_type") == "over_translation" for row in null_rows
            ),
            "under_translation_alignments": sum(
                row.get("null_alignment_type") == "under_translation" for row in null_rows
            ),
            "null_alignment_ratio": len(null_rows) / len(run_rows) if run_rows else 0.0,
            "null_alignment_score": 0.0,
            "aggregation": "arithmetic_mean_including_null_alignment_zeros",
            "output_jsonl": str(out_dir / "segments.jsonl"),
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_jsonl)
    scoreable_indices = [
        index for index, row in enumerate(rows) if row.get("fixed_xcomet_xl_score") is None
    ]
    model_scores: list[float] = []
    if scoreable_indices:
        configure_comet_prediction_load_compatibility()
        from comet import download_model, load_from_checkpoint

        model_path = download_model(args.model_name)
        model = load_from_checkpoint(model_path)
        scoreable_rows = [rows[index] for index in scoreable_indices]
        output = model.predict(
            comet_data(scoreable_rows, args.reference_free),
            batch_size=args.batch_size,
            gpus=args.gpus,
        )
        model_scores = [float(score) for score in output.scores]
        if len(model_scores) != len(scoreable_rows):
            raise RuntimeError(f"expected {len(scoreable_rows)} scores, got {len(model_scores)}")

    mode = "source_hypothesis" if args.reference_free else "source_hypothesis_reference"
    out_rows = attach_scores(rows, model_scores, args.model_name, mode)
    write_jsonl(args.output_jsonl, out_rows)
    if args.write_run_summaries:
        write_run_summaries(out_rows, args.model_name, mode)
    system_score = arithmetic_mean(out_rows, "xcomet_xl_score")
    null_rows = [row for row in out_rows if row.get("null_alignment_type")]
    summary = {
        "xcomet_xl": system_score,
        "xcomet_xl_model": args.model_name,
        "xcomet_xl_mode": mode,
        "alignment_backend": "SEGALE",
        "segments": len(out_rows),
        "valid_segments": len(out_rows) - len(null_rows),
        "null_alignments": len(null_rows),
        "null_alignment_ratio": len(null_rows) / len(out_rows) if out_rows else 0.0,
        "null_alignment_score": 0.0,
        "aggregation": "arithmetic_mean_including_null_alignment_zeros",
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
