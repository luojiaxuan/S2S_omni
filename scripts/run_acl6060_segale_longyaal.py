#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from build_acl6060_segale_inputs import resolve_path
from run_acl6060_segale_alignment import PINNED_SPEECH_LATENCY_REVISION, git_revision

BLEU_TOKENIZER_BY_LANG = {"zh": "zh", "de": "13a", "ja": "ja-mecab"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SEGALE-based LongYAAL for one ACL6060 run.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--speech-latency-repo", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--expected-revision", default=PINNED_SPEECH_LATENCY_REVISION)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_scores(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        row = next(reader)
    return {key: float(value) for key, value in row.items() if key and value}


def run_longyaal(args: argparse.Namespace) -> dict[str, Any]:
    alignment_dir = args.run_dir / "segale_alignment"
    output_dir = args.output_dir or (args.run_dir / "segale_longyaal")
    summary_path = output_dir / "summary.json"
    scores_path = output_dir / "scores.resegmented.csv"
    if args.resume and summary_path.exists() and scores_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    revision = git_revision(args.speech_latency_repo)
    if args.expected_revision and revision != args.expected_revision:
        raise RuntimeError(
            f"Speech-to-Speech-Latency revision mismatch: {revision} != {args.expected_revision}"
        )
    config = json.loads((args.run_dir / "run_config.json").read_text(encoding="utf-8"))
    dataset_root = args.dataset_root
    if dataset_root is None and config.get("dataset_root"):
        dataset_root = Path(str(config["dataset_root"]))
    source_text = resolve_path(config["source_text_file"], dataset_root)
    target_lang = str(config["target_lang"])

    sys.path.insert(0, str(args.speech_latency_repo))
    from src.evaluation.eval import step3_longyaal

    output_dir.mkdir(parents=True, exist_ok=True)
    step3_longyaal(
        yaml_file=str(alignment_dir / "audio.scaled.basename.yaml"),
        source_sentences_file=str(source_text),
        instances_log=str(alignment_dir / "instances.segale.jsonl"),
        segale_file=str(alignment_dir / "hyp" / "aligned_spacy_hyp.jsonl"),
        output_folder=str(output_dir),
        bleu_tokenizer=BLEU_TOKENIZER_BY_LANG[target_lang],
    )
    scores = read_scores(scores_path)
    quality = json.loads((alignment_dir / "quality_summary.json").read_text(encoding="utf-8"))
    summary = {
        "alignment_backend": "SEGALE",
        "speech_latency_revision": revision,
        "bleu": quality["bleu"],
        "bleu_tokenizer": quality["bleu_tokenizer"],
        "longyaal_cu": scores.get("ca_unaware_yaal"),
        "longyaal_ca": scores.get("ca_aware_yaal"),
        "ending_offset_cu_ms_mean": scores.get("ending_offset"),
        "ending_offset_ca_ms_mean": scores.get("ca_ending_offset"),
        "latency_valid_segments": sum(
            1
            for row in json.loads(
                (output_dir / "instances.resegmented.json").read_text(encoding="utf-8")
            )
            if row.get("delays")
        ),
        "null_alignments": quality["null_alignments"],
        "null_alignment_ratio": quality["null_alignment_ratio"],
        "scores_path": str(scores_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(run_longyaal(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
