#!/usr/bin/env python3
# ruff: noqa: E501
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sentence-level SEGALE, XCOMET, and latency diagnostics for ACL6060."
    )
    parser.add_argument(
        "--artifact-base",
        type=Path,
        default=Path("projects/acl6060_s2s_metrics_seed/artifacts"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def number(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def escaped(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def latency_fields(row: dict[str, Any] | None) -> dict[str, float | None]:
    if row is None:
        return {
            "first_emission_offset_ms": None,
            "tail_latency_ms": None,
            "emission_span_ms": None,
            "target_units": None,
        }
    elapsed = [float(value) for value in row.get("elapsed") or []]
    source_length = float(row.get("source_length") or 0.0)
    if not elapsed:
        return {
            "first_emission_offset_ms": None,
            "tail_latency_ms": None,
            "emission_span_ms": None,
            "target_units": len(row.get("raw_units") or []),
        }
    return {
        "first_emission_offset_ms": elapsed[0] - source_length,
        "tail_latency_ms": elapsed[-1] - source_length,
        "emission_span_ms": elapsed[-1] - elapsed[0],
        "target_units": len(row.get("raw_units") or []),
    }


def source_key(document: str, segment_id: int) -> tuple[str, int]:
    return str(document), int(segment_id)


def build_run(
    run_dir: Path, table_row: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = read_json(run_dir / "run_config.json")
    quality = read_json(run_dir / "segale_alignment" / "quality_summary.json")
    xcomet_rows = read_jsonl(run_dir / "xcomet_xl" / "segments.jsonl")
    latency_rows = read_json(run_dir / "segale_longyaal" / "instances.resegmented.json")
    source_rows = read_jsonl(run_dir / "segale_alignment" / "ref.jsonl")
    if not source_rows:
        raise ValueError(f"empty SEGALE reference rows: {run_dir}")
    latency_by_key = {
        source_key(row["doc_id"], int(row["seg_id"])): row for row in latency_rows
    }
    sentence_rows: list[dict[str, Any]] = []
    expected_keys = set()
    for row in xcomet_rows:
        document = str(row["doc_id"])
        segment_id = int(row["segale_segment_id"])
        key = source_key(document, segment_id)
        expected_keys.add(key)
        null_type = str(row.get("null_alignment_type") or "")
        latency = latency_fields(latency_by_key.get(key))
        if null_type:
            trace = latency_by_key.get(key)
            if trace is None:
                raise ValueError(f"null alignment missing sentinel trace: {run_dir} {key}")
            if trace.get("elapsed") or trace.get("source_length") is not None:
                raise ValueError(f"null alignment has timing data: {run_dir} {key}")
        else:
            if key not in latency_by_key:
                raise ValueError(f"valid alignment missing latency: {run_dir} {key}")
            if latency["tail_latency_ms"] is None:
                raise ValueError(f"valid alignment missing timing data: {run_dir} {key}")
        sentence_rows.append(
            {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "language": table_row["Language"],
                "provider": str(config["provider"]),
                "system": table_row["System"],
                "target_lang": str(config["target_lang"]),
                "speed_factor": float(config["speed_factor"]),
                "doc_id": document,
                "segale_segment_id": segment_id,
                "source_segment_ids": [int(value) for value in row["source_segment_ids"]],
                "hypothesis_sentence_ids": [int(value) for value in row["hypothesis_sentence_ids"]],
                "alignment_shape": (
                    f"{len(row['source_segment_ids'])}:{len(row['hypothesis_sentence_ids'])}"
                ),
                "source": row["source"],
                "reference": row["reference"],
                "hypothesis": row["hypothesis"],
                "null_alignment_type": null_type,
                "xcomet_xl_score": float(row["xcomet_xl_score"]),
                "xcomet_xl_score_source": row["xcomet_xl_score_source"],
                **latency,
            }
        )
    if expected_keys != set(latency_by_key):
        extra = sorted(set(latency_by_key) - expected_keys)[:3]
        missing = sorted(expected_keys - set(latency_by_key))[:3]
        raise ValueError(
            f"latency alignment mismatch for {run_dir}: extra={extra} missing={missing}"
        )

    non_null = [row for row in sentence_rows if not row["null_alignment_type"]]
    tails = [
        float(row["tail_latency_ms"])
        for row in non_null
        if row["tail_latency_ms"] is not None
    ]
    firsts = [
        float(row["first_emission_offset_ms"])
        for row in non_null
        if row["first_emission_offset_ms"] is not None
    ]
    spans = [
        float(row["emission_span_ms"])
        for row in non_null
        if row["emission_span_ms"] is not None
    ]
    over = sum(row["null_alignment_type"] == "over_translation" for row in sentence_rows)
    under = sum(row["null_alignment_type"] == "under_translation" for row in sentence_rows)
    summary = {
        "Language": table_row["Language"],
        "Speedup": table_row["Speedup"],
        "System": table_row["System"],
        "run_dir": str(run_dir),
        "provider": config["provider"],
        "target_lang": config["target_lang"],
        "speed_factor": float(config["speed_factor"]),
        "bleu": float(quality["bleu"]),
        "xcomet_xl": mean(float(row["xcomet_xl_score"]) for row in sentence_rows),
        "segments": len(sentence_rows),
        "valid_segments": len(non_null),
        "over_translation_alignments": over,
        "under_translation_alignments": under,
        "null_alignments": over + under,
        "tail_latency_mean_ms": mean(tails) if tails else None,
        "tail_latency_p50_ms": percentile(tails, 0.5),
        "tail_latency_p90_ms": percentile(tails, 0.9),
        "first_emission_offset_p50_ms": percentile(firsts, 0.5),
        "emission_span_p50_ms": percentile(spans, 0.5),
    }
    return summary, sentence_rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "Language",
        "Speedup",
        "System",
        "provider",
        "target_lang",
        "speed_factor",
        "run_dir",
        "bleu",
        "xcomet_xl",
        "segments",
        "valid_segments",
        "over_translation_alignments",
        "under_translation_alignments",
        "null_alignments",
        "tail_latency_mean_ms",
        "tail_latency_p50_ms",
        "tail_latency_p90_ms",
        "first_emission_offset_p50_ms",
        "emission_span_p50_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_speed_delta_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "Language",
        "System",
        "provider",
        "target_lang",
        "source_sentences_all_speeds",
        "valid_pairs_1p25x_vs_1x",
        "xcomet_delta_mean_1p25x_vs_1x",
        "tail_delta_mean_ms_1p25x_vs_1x",
        "tail_delta_p50_ms_1p25x_vs_1x",
        "tail_delta_p90_ms_1p25x_vs_1x",
        "first_delta_p50_ms_1p25x_vs_1x",
        "valid_pairs_1p5x_vs_1x",
        "xcomet_delta_mean_1p5x_vs_1x",
        "tail_delta_mean_ms_1p5x_vs_1x",
        "tail_delta_p50_ms_1p5x_vs_1x",
        "tail_delta_p90_ms_1p5x_vs_1x",
        "first_delta_p50_ms_1p5x_vs_1x",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_index(summary_rows: list[dict[str, Any]], output_dir: Path) -> str:
    rows = []
    for row in summary_rows:
        page = f"compare_{row['target_lang']}_{row['provider']}.html"
        rows.append(
            "<tr>"
            f"<td>{escaped(row['Language'])}</td><td>{escaped(row['System'])}</td>"
            f"<td>{escaped(row['Speedup'])}</td><td>{number(row['bleu'], 4)}</td>"
            f"<td>{number(row['xcomet_xl'], 6)}</td><td>{row['valid_segments']}/{row['segments']}</td>"
            f"<td>{row['over_translation_alignments']}</td><td>{row['under_translation_alignments']}</td>"
            f"<td>{number(row['tail_latency_mean_ms'])}</td>"
            f"<td>{number(row['tail_latency_p50_ms'])}</td>"
            f"<td>{number(row['tail_latency_p90_ms'])}</td>"
            f"<td>{number(row['first_emission_offset_p50_ms'])}</td>"
            f"<td>{number(row['emission_span_p50_ms'])}</td>"
            f"<td><a href='{escaped(page)}'>sentence cases</a></td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>ACL6060 SEGALE diagnostics</title>
<style>
body{{font-family:Arial,sans-serif;margin:28px;color:#20242a;background:#fafbfc}}h1{{margin-bottom:8px}}
p{{max-width:1200px;line-height:1.45}}table{{width:100%;border-collapse:collapse;background:#fff;font-size:13px}}
th,td{{border:1px solid #d8dde3;padding:7px;text-align:right;vertical-align:top}}th{{background:#edf1f5;position:sticky;top:0}}th:nth-child(-n+3),td:nth-child(-n+3){{text-align:left}}a{{color:#075985}}
.note{{background:#fff8db;border-left:4px solid #ca8a04;padding:10px;max-width:1200px}}
</style></head><body>
<h1>ACL6060 SEGALE sentence diagnostics</h1>
<p class='note'><strong>Latency semantics:</strong> tail and first-emission offsets use the arrival time of target transcript deltas. They are a computation-aware target-text proxy, not verified target-audio playback completion: the formal GPT/Gemini runs did not save output audio timestamps.</p>
<p>Each sentence-case page compares the same ACL source sentence across source speeds. A SEGALE row can be many-to-many; its shape is shown as source:hypothesis. Null rows are retained with XCOMET=0.0 but have no latency because no target arrival exists.</p>
<table><thead><tr><th>language</th><th>system</th><th>speed</th><th>BLEU</th><th>XCOMET</th><th>valid/all</th><th>over</th><th>under</th><th>tail mean ms</th><th>tail p50 ms</th><th>tail p90 ms</th><th>first p50 ms</th><th>emit span p50 ms</th><th>details</th></tr></thead><tbody>
{''.join(rows)}</tbody></table>
<p>Machine-readable summaries: <a href='cell_summary.tsv'>per-cell TSV</a>, <a href='cell_summary.jsonl'>per-cell JSONL</a>, <a href='speed_delta_summary.tsv'>paired-speed TSV</a>, <a href='speed_delta_summary.jsonl'>paired-speed JSONL</a>, <a href='sentence_cases.jsonl'>all source-sentence cases</a>.</p>
</body></html>"""


def render_speed_page(
    language: str,
    system: str,
    provider: str,
    target_lang: str,
    by_speed: dict[float, dict[int, dict[str, Any]]],
) -> str:
    all_source_ids = sorted({source_id for rows in by_speed.values() for source_id in rows})
    entries = []
    for source_id in all_source_ids:
        candidates = [rows.get(source_id) for rows in by_speed.values() if source_id in rows]
        anchor = next(row for row in candidates if row is not None)
        rows = []
        summary_bits = []
        for speed in sorted(by_speed):
            row = by_speed[speed].get(source_id)
            if row is None:
                rows.append(f"<tr><td>{speed:g}x</td><td colspan='7'>no aligned row</td></tr>")
                continue
            null_type = row["null_alignment_type"] or "matched"
            summary_bits.append(
                f"{speed:g}x: {null_type}, QE {number(row['xcomet_xl_score'], 3)}, tail {number(row['tail_latency_ms'])} ms"
            )
            rows.append(
                "<tr>"
                f"<td>{speed:g}x</td><td>{escaped(row['alignment_shape'])}</td>"
                f"<td>{escaped(null_type)}</td><td>{number(row['xcomet_xl_score'], 6)}</td>"
                f"<td>{number(row['first_emission_offset_ms'])}</td>"
                f"<td>{number(row['tail_latency_ms'])}</td><td>{number(row['emission_span_ms'])}</td>"
                f"<td>{escaped(row['hypothesis'])}</td></tr>"
            )
        entries.append(
            "<details><summary>"
            f"{escaped(anchor['doc_id'])} source #{source_id} | {'; '.join(summary_bits)}"
            "</summary>"
            f"<p><strong>Source:</strong> {escaped(anchor['source_sentence'])}</p>"
            f"<p><strong>Reference:</strong> {escaped(anchor['reference_sentence'])}</p>"
            "<table><thead><tr><th>speed</th><th>shape</th><th>alignment</th><th>XCOMET</th>"
            "<th>first offset ms</th><th>tail offset ms</th><th>emit span ms</th><th>hypothesis</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></details>"
        )
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{escaped(language)} {escaped(system)} speed cases</title>
<style>
body{{font-family:Arial,sans-serif;margin:28px;color:#20242a;background:#fafbfc}}a{{color:#075985}}p{{line-height:1.45}}
details{{margin:10px 0;background:#fff;border:1px solid #d8dde3;border-radius:6px;padding:10px}}summary{{cursor:pointer;font-weight:600;line-height:1.45}}
table{{width:100%;border-collapse:collapse;font-size:13px;table-layout:fixed}}th,td{{border:1px solid #d8dde3;padding:7px;vertical-align:top;text-align:right;word-break:break-word}}th{{background:#edf1f5}}th:last-child,td:last-child{{text-align:left;width:38%}}
.note{{background:#fff8db;border-left:4px solid #ca8a04;padding:10px}}
</style></head><body>
<p><a href='index.html'>&larr; all cells</a></p><h1>{escaped(language)} | {escaped(system)} | source-speed cases</h1>
<p class='note'>First and tail offsets are target transcript-delta arrival minus source sentence end. Negative values mean the text arrived before that source sentence ended. This is not target-audio playback latency. Shape is SEGALE source:hypothesis; a multi-source row intentionally repeats the same aligned hypothesis under each source sentence it covers.</p>
{''.join(entries)}
</body></html>"""


def paired_delta_row(
    language: str,
    system: str,
    provider: str,
    target_lang: str,
    by_speed: dict[float, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    speeds = (1.0, 1.25, 1.5)
    if any(speed not in by_speed for speed in speeds):
        raise ValueError(f"missing required source speed: {language} {provider}")
    all_sources = set.intersection(*(set(by_speed[speed]) for speed in speeds))
    result: dict[str, Any] = {
        "Language": language,
        "System": system,
        "provider": provider,
        "target_lang": target_lang,
        "source_sentences_all_speeds": len(all_sources),
    }
    for speed, label in ((1.25, "1p25x"), (1.5, "1p5x")):
        pairs = []
        for source_id in all_sources:
            baseline = by_speed[1.0][source_id]
            comparison = by_speed[speed][source_id]
            if (
                baseline["null_alignment_type"]
                or comparison["null_alignment_type"]
                or baseline["tail_latency_ms"] is None
                or comparison["tail_latency_ms"] is None
                or baseline["first_emission_offset_ms"] is None
                or comparison["first_emission_offset_ms"] is None
            ):
                continue
            pairs.append((baseline, comparison))
        xcomet_deltas = [
            float(comparison["xcomet_xl_score"]) - float(baseline["xcomet_xl_score"])
            for baseline, comparison in pairs
        ]
        tail_deltas = [
            float(comparison["tail_latency_ms"]) - float(baseline["tail_latency_ms"])
            for baseline, comparison in pairs
        ]
        first_deltas = [
            float(comparison["first_emission_offset_ms"])
            - float(baseline["first_emission_offset_ms"])
            for baseline, comparison in pairs
        ]
        result[f"valid_pairs_{label}_vs_1x"] = len(pairs)
        result[f"xcomet_delta_mean_{label}_vs_1x"] = mean(xcomet_deltas) if pairs else None
        result[f"tail_delta_mean_ms_{label}_vs_1x"] = mean(tail_deltas) if pairs else None
        result[f"tail_delta_p50_ms_{label}_vs_1x"] = percentile(tail_deltas, 0.5)
        result[f"tail_delta_p90_ms_{label}_vs_1x"] = percentile(tail_deltas, 0.9)
        result[f"first_delta_p50_ms_{label}_vs_1x"] = percentile(first_deltas, 0.5)
    return result


def build_comparison_pages(
    sentence_rows: list[dict[str, Any]], output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sentence_rows:
        grouped[(row["language"], row["system"], row["provider"], row["target_lang"])].append(row)
    delta_rows = []
    source_sentence_cases = []
    for (language, system, provider, target_lang), rows in grouped.items():
        by_speed: dict[float, dict[int, dict[str, Any]]] = defaultdict(dict)
        references: dict[int, dict[str, Any]] = {}
        for row in rows:
            run_dir = Path(row["run_dir"])
            ref_rows = read_jsonl(run_dir / "segale_alignment" / "ref.jsonl")
            for source_row in ref_rows:
                source_id = int(source_row["seg_id"])
                references[source_id] = {
                    "source_sentence": source_row["src"],
                    "reference_sentence": source_row["tgt"],
                }
            for source_id in row["source_segment_ids"]:
                if source_id in by_speed[row["speed_factor"]]:
                    raise ValueError(f"duplicate source case: {run_dir} {source_id}")
                source_case = {
                    **row,
                    "source_segment_id": source_id,
                    **references[source_id],
                }
                by_speed[row["speed_factor"]][source_id] = source_case
                source_sentence_cases.append(source_case)
        page = render_speed_page(language, system, provider, target_lang, by_speed)
        (output_dir / f"compare_{target_lang}_{provider}.html").write_text(
            page, encoding="utf-8"
        )
        delta_rows.append(
            paired_delta_row(language, system, provider, target_lang, by_speed)
        )
    source_sentence_cases.sort(
        key=lambda row: (
            row["language"],
            row["system"],
            row["speed_factor"],
            row["doc_id"],
            row["source_segment_id"],
        )
    )
    return delta_rows, source_sentence_cases


def main() -> None:
    args = parse_args()
    table = read_jsonl(args.artifact_base / "acl6060_full_table.jsonl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    sentence_rows = []
    for row in table:
        run_dir = args.artifact_base / Path(str(row["run_dir"])).name
        summary, run_sentences = build_run(run_dir, row)
        summaries.append(summary)
        sentence_rows.extend(run_sentences)
    summaries.sort(key=lambda row: (row["Language"], row["System"], row["speed_factor"]))
    write_jsonl(args.output_dir / "cell_summary.jsonl", summaries)
    write_tsv(args.output_dir / "cell_summary.tsv", summaries)
    delta_rows, source_sentence_cases = build_comparison_pages(sentence_rows, args.output_dir)
    write_jsonl(args.output_dir / "sentence_cases.jsonl", source_sentence_cases)
    delta_rows.sort(key=lambda row: (row["Language"], row["System"]))
    write_jsonl(args.output_dir / "speed_delta_summary.jsonl", delta_rows)
    write_speed_delta_tsv(args.output_dir / "speed_delta_summary.tsv", delta_rows)
    (args.output_dir / "index.html").write_text(
        render_index(summaries, args.output_dir), encoding="utf-8"
    )
    manifest = {
        "cells": len(summaries),
        "alignment_groups": len(sentence_rows),
        "source_sentence_cases": len(source_sentence_cases),
        "null_alignments": sum(int(row["null_alignments"]) for row in summaries),
        "output_dir": str(args.output_dir),
        "latency_semantics": "target transcript-delta arrival proxy, not target-audio playback",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
