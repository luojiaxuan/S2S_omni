#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MetricX-24 reference-free QE over FLORAS QE segments.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-name", default="google/metricx-24-hybrid-large-v2p6-bfloat16")
    parser.add_argument("--tokenizer", default="google/mt5-large")
    parser.add_argument("--metricx-code-dir", default="")
    parser.add_argument("--max-input-length", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def make_input(row: dict[str, Any]) -> str:
    return "source: " + str(row["source"]) + " candidate: " + str(row["hypothesis"])


def tokenize_batch(tokenizer: Any, texts: list[str], max_input_length: int) -> dict[str, Any]:
    features = []
    for text in texts:
        feature = tokenizer(text, max_length=max_input_length, truncation=True, padding=False)
        feature["input_ids"] = feature["input_ids"][:-1]
        feature["attention_mask"] = feature["attention_mask"][:-1]
        features.append(feature)
    return tokenizer.pad(features, return_tensors="pt")


def main() -> None:
    args = parse_args()
    if args.metricx_code_dir:
        sys.path.insert(0, str(Path(args.metricx_code_dir).expanduser()))

    import torch
    import transformers
    from metricx24 import models

    rows = read_jsonl(Path(args.input_jsonl).expanduser())
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    model = models.MT5ForRegression.from_pretrained(args.model_name, torch_dtype="auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    out_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            batch = tokenize_batch(tokenizer, [make_input(row) for row in batch_rows], args.max_input_length)
            batch = {key: value.to(device) for key, value in batch.items()}
            predictions = model(**batch, use_cache=False).predictions.detach().cpu().tolist()
            for row, prediction in zip(batch_rows, predictions):
                out = dict(row)
                out["prediction"] = float(prediction)
                out["metricx_qe_model"] = args.model_name
                out["metricx_qe_mode"] = "source_hypothesis_no_reference"
                out_rows.append(out)
    write_jsonl(Path(args.output_jsonl).expanduser(), out_rows)
    print(json.dumps({"segments": len(out_rows), "output": args.output_jsonl}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
