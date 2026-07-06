#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run xCOMET-style reference-free QE over FLORAS QE segments.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-name", default="myyycroft/XCOMET-lite")
    parser.add_argument("--xcomet-code-dir", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--include-empty-ref", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.xcomet_code_dir:
        sys.path.insert(0, str(Path(args.xcomet_code_dir).expanduser()))
    from xcomet.deberta_encoder import XCOMETLite

    rows = read_jsonl(Path(args.input_jsonl).expanduser())
    data = []
    for row in rows:
        item = {"src": row["source"], "mt": row["hypothesis"]}
        if args.include_empty_ref:
            item["ref"] = ""
        data.append(item)
    model = XCOMETLite().from_pretrained(args.model_name)
    output = model.predict(data, batch_size=args.batch_size, gpus=args.gpus)
    scores = [float(score) for score in output.scores]
    if len(scores) != len(rows):
        raise SystemExit(f"expected {len(rows)} xCOMET scores, got {len(scores)}")
    out_rows = []
    for row, score in zip(rows, scores):
        out_rows.append(
            {
                "qe_id": row["qe_id"],
                "qe_row_key": row["qe_row_key"],
                "segment_index": row["segment_index"],
                "segment_count": row["segment_count"],
                "weight_chars": row["weight_chars"],
                "xcomet_qe_score": round(score, 6),
                "xcomet_qe_model": args.model_name,
                "xcomet_qe_mode": "source_hypothesis_no_reference",
            }
        )
    write_jsonl(Path(args.output_jsonl).expanduser(), out_rows)
    print(json.dumps({"segments": len(out_rows), "output": args.output_jsonl}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
