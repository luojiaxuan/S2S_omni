#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl, write_jsonl
from s2s_omni.judge import combine_judge_and_heuristics, judge_sample
from s2s_omni.llm_client import ChatClient
from s2s_omni.metrics import (
    heuristic_score_sample,
    optional_sacrebleu,
    summarize_metric_rows,
)
from s2s_omni.schema import S2SSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate compressed S2ST predictions.")
    parser.add_argument("--gold", required=True, help="Gold manifest JSONL.")
    parser.add_argument("--pred", required=True, help="Prediction JSONL.")
    parser.add_argument("--output", required=True, help="Per-sample metrics JSONL.")
    parser.add_argument("--summary", default=None, help="Optional summary JSON path.")
    parser.add_argument("--judge", action="store_true", help="Use LLM-as-judge.")
    parser.add_argument("--prediction-field", default="candidate_translation")
    parser.add_argument(
        "--skip-sacrebleu",
        action="store_true",
        help="Do not compute corpus BLEU/chrF even if sacrebleu is installed.",
    )
    return parser.parse_args()


def load_prediction_map(path: str, field: str) -> dict[str, str]:
    records = read_jsonl(path)
    out: dict[str, str] = {}
    for record in records:
        sample_id = str(record.get("id") or record.get("sample_id") or "")
        if not sample_id:
            continue
        value = (
            record.get(field)
            or record.get("candidate_translation")
            or record.get("compressed_translation")
            or record.get("prediction")
            or record.get("text")
        )
        if value is not None:
            out[sample_id] = str(value)
    return out


def load_eval_samples(path: str) -> list[S2SSample]:
    samples: list[S2SSample] = []
    for record in read_jsonl(path):
        payload = record.get("sample") if isinstance(record.get("sample"), dict) else record
        samples.append(S2SSample.from_dict(payload))
    return samples


def main() -> None:
    args = parse_args()
    gold = load_eval_samples(args.gold)
    predictions = load_prediction_map(args.pred, args.prediction_field)
    judge_client = ChatClient.from_env() if args.judge else None

    rows = []
    missing = []
    bleu_candidates: list[str] = []
    bleu_references: list[str] = []
    for sample in gold:
        candidate = predictions.get(sample.id)
        if candidate is None and "__speed_" in sample.id:
            candidate = predictions.get(sample.metadata.get("base_id", ""))
        if candidate is None:
            missing.append(sample.id)
            continue
        heuristic = heuristic_score_sample(sample, candidate)
        judge = judge_sample(sample, candidate, judge_client) if judge_client else None
        rows.append(combine_judge_and_heuristics(heuristic, judge))
        reference = sample.reference_translation or sample.compressed_translation
        if reference:
            bleu_candidates.append(candidate)
            bleu_references.append(reference)

    write_jsonl(args.output, rows)
    summary = summarize_metric_rows(rows)
    summary["missing_predictions"] = missing
    if not args.skip_sacrebleu:
        summary["corpus_text_quality"] = optional_sacrebleu(
            bleu_candidates, bleu_references
        )
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"scored {len(rows)} rows; missing {len(missing)}; wrote {args.output}")
    if args.summary:
        print(f"summary: {args.summary}")


if __name__ == "__main__":
    main()
