#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

BLEU_TOKENIZER_BY_LANG = {"zh": "zh", "de": "13a", "ja": "ja-mecab"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ACL6060 XCOMET-XL input from SEGALE source-hypothesis alignments."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--aligned-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--bleu-tokenizer", default="")
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


def read_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_key(run_dir: Path, config: dict[str, Any]) -> str:
    provider = str(config.get("provider") or "")
    target_lang = str(config.get("target_lang") or "")
    chunk_ms = str(config.get("chunk_ms") or "")
    speed = str(config.get("speed_factor") or "")
    return "||".join([run_dir.name, provider, target_lang, chunk_ms, speed])


def null_alignment_type(source: str, hypothesis: str) -> str:
    if not source.strip():
        return "over_translation"
    if not hypothesis.strip():
        return "under_translation"
    return ""


def validate_alignment_coverage(
    segments: list[dict[str, Any]], expected_source_segments: int
) -> None:
    source_ids = [
        int(source_id)
        for segment in segments
        for source_id in list(segment.get("src_ref_ids") or [])
    ]
    if sorted(source_ids) != list(range(1, expected_source_segments + 1)):
        raise ValueError(
            "SEGALE source coverage mismatch: "
            f"expected 1..{expected_source_segments}, got {len(source_ids)} ids"
        )

    target_ids: dict[str, list[int]] = defaultdict(list)
    for segment in segments:
        target_ids[str(segment["doc_id"])].extend(
            int(value) for value in segment.get("mt_indices") or []
        )
    for document, values in target_ids.items():
        if sorted(values) != list(range(len(values))):
            raise ValueError(f"SEGALE target coverage mismatch for {document}: {values}")


def corpus_bleu(rows: list[dict[str, Any]], tokenizer: str) -> float:
    from sacrebleu import corpus_bleu as sacrebleu_corpus_bleu

    return float(
        sacrebleu_corpus_bleu(
            [str(row["hypothesis"]) for row in rows],
            [[str(row["reference"]) for row in rows]],
            tokenize=tokenizer,
        ).score
    )


def build_xcomet_rows(
    run_dir: Path,
    segments: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    key = run_key(run_dir, config)
    rows = []
    for index, segment in enumerate(segments):
        source = str(segment.get("src") or "")
        hypothesis = str(segment.get("tgt") or "")
        reference = str(segment.get("ref") or "")
        null_type = null_alignment_type(source, hypothesis)
        if null_type == "over_translation" and reference.strip():
            raise ValueError(f"over-translation row {index} unexpectedly has a reference")
        rows.append(
            {
                "xcomet_id": f"{key}||segale||{index:04d}",
                "run_key": key,
                "run_dir": str(run_dir),
                "segment_index": index,
                "doc_id": segment.get("doc_id"),
                "segale_segment_id": segment.get("seg_id"),
                "source_segment_ids": list(segment.get("src_ref_ids") or []),
                "hypothesis_sentence_ids": list(segment.get("mt_indices") or []),
                "source": source,
                "hypothesis": hypothesis,
                "reference": reference,
                "null_alignment_type": null_type,
                "fixed_xcomet_xl_score": 0.0 if null_type else None,
                "alignment_backend": "SEGALE",
                "target_lang": config.get("target_lang"),
                "provider": config.get("provider"),
                "chunk_ms": config.get("chunk_ms"),
                "speed_factor": config.get("speed_factor"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = read_config(args.run_dir)
    alignment_dir = args.run_dir / "segale_alignment"
    aligned_jsonl = args.aligned_jsonl or (alignment_dir / "hyp" / "aligned_spacy_hyp.jsonl")
    summary_json = args.summary_json or (alignment_dir / "quality_summary.json")
    input_summary = json.loads((alignment_dir / "input_summary.json").read_text(encoding="utf-8"))
    alignment_summary = json.loads(
        (alignment_dir / "alignment_summary.json").read_text(encoding="utf-8")
    )
    segments = read_jsonl(aligned_jsonl)
    validate_alignment_coverage(segments, int(input_summary["source_segments"]))
    rows = build_xcomet_rows(args.run_dir, segments, config)
    tokenizer = args.bleu_tokenizer or BLEU_TOKENIZER_BY_LANG[str(config["target_lang"])]
    over_count = sum(row["null_alignment_type"] == "over_translation" for row in rows)
    under_count = sum(row["null_alignment_type"] == "under_translation" for row in rows)
    null_count = over_count + under_count
    write_jsonl(args.output_jsonl, rows)
    summary = {
        "alignment_backend": "SEGALE",
        "speech_latency_revision": alignment_summary.get("speech_latency_revision"),
        "embedding_model": alignment_summary.get("embedding_model"),
        "segmenter": alignment_summary.get("segmenter"),
        "max_size": alignment_summary.get("max_size"),
        "segments": len(rows),
        "valid_segments": len(rows) - null_count,
        "null_alignments": null_count,
        "over_translation_alignments": over_count,
        "under_translation_alignments": under_count,
        "null_alignment_ratio": null_count / len(rows) if rows else 0.0,
        "null_alignment_score": 0.0,
        "bleu": corpus_bleu(rows, tokenizer),
        "bleu_tokenizer": tokenizer,
        "aligned_jsonl": str(aligned_jsonl),
        "xcomet_input_jsonl": str(args.output_jsonl),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
