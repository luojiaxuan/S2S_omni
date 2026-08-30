#!/usr/bin/env python3
"""Repack prepared MOSS-TTS rows to the InfiniSST phrase-policy turn grid.

The source prepared rows were encoded from whole-passage audio and then sliced
at the original trajectory boundaries. This script replays the exact phrase
redistribution used by InfiniSST, merges adjacent text and codec slices, and
keeps the concatenated text and codec frames bit-identical per row.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import statistics
from pathlib import Path
from typing import Any


PHRASE_PUNCT = "。！？!?…，,、；;：:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-tsv", required=True, type=Path)
    parser.add_argument("--prepared-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--stats-json", required=True, type=Path)
    parser.add_argument("--id-prefix", default="traj_")
    parser.add_argument("--multiplier", type=int, default=2)
    parser.add_argument("--chunk-seconds", type=float, default=0.96)
    parser.add_argument("--phrase-max-hold-s", type=float, default=7.68)
    parser.add_argument("--phrase-min-chars", type=int, default=6)
    return parser.parse_args()


def read_trajectories(path: Path) -> dict[str, list[str]]:
    csv.field_size_limit(10**8)
    trajectories: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            raw = (row.get("trajectory") or "").lstrip()
            if not raw.startswith("["):
                continue
            try:
                trajectory = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(trajectory, list) or not all(
                isinstance(text, str) for text in trajectory
            ):
                continue
            trajectories[str(row["id"])] = trajectory
    return trajectories


def phrase_redistribute(
    trajectory: list[str], multiplier: int, max_hold_steps: int, min_chars: int
) -> list[str]:
    grouped = [
        "".join(trajectory[index : index + multiplier])
        for index in range(0, len(trajectory), multiplier)
    ]
    outputs: list[str] = []
    buffer = ""
    held = 0
    for index, text in enumerate(grouped):
        buffer += text
        stripped = buffer.strip()
        if not stripped:
            held = 0
            continue
        held += 1
        spoken_chars = sum(char.isalnum() for char in buffer)
        at_boundary = stripped[-1] in PHRASE_PUNCT and spoken_chars >= min_chars
        if at_boundary or held >= max_hold_steps or index == len(grouped) - 1:
            outputs.append(buffer)
            buffer = ""
            held = 0
    if buffer:
        if not outputs:
            outputs.append(buffer)
        else:
            outputs[-1] += buffer
    return outputs


def cumulative_spans(texts: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    offset = 0
    for text in texts:
        spans.append((offset, offset + len(text)))
        offset += len(text)
    return spans


def repack_row(
    row: dict[str, Any], new_texts: list[str]
) -> tuple[dict[str, Any], int, int]:
    old_turns = row["conversations"]
    old_texts = [str(turn["text"]) for turn in old_turns]
    if "".join(old_texts) != "".join(new_texts):
        raise ValueError(f"{row['id']}: concatenated text changed")

    new_spans = cumulative_spans(new_texts)
    new_codes: list[list[list[int]]] = [[] for _ in new_texts]
    punctuation_splits = 0
    previous_owner = -1
    old_offset = 0
    for old_turn, old_text in zip(old_turns, old_texts, strict=True):
        old_start = old_offset
        old_end = old_start + len(old_text)
        old_offset = old_end
        overlaps: list[tuple[int, int, int]] = []
        for new_index, (new_start, new_end) in enumerate(new_spans):
            start = max(old_start, new_start)
            end = min(old_end, new_end)
            if start >= end:
                continue
            substring = old_text[start - old_start : end - old_start]
            overlaps.append(
                (new_index, sum(char.isalnum() for char in substring), end - start)
            )
        if not overlaps:
            raise ValueError(f"{row['id']}: old turn has no phrase overlap")
        spoken_owners = [item for item in overlaps if item[1] > 0]
        if len(spoken_owners) > 1:
            raise ValueError(f"{row['id']}: phrase boundary splits spoken text")
        if spoken_owners:
            owner = spoken_owners[0][0]
        else:
            owner = max(overlaps, key=lambda item: item[2])[0]
        if len(overlaps) > 1:
            punctuation_splits += 1
        if owner < previous_owner:
            raise ValueError(f"{row['id']}: codec ownership is not monotonic")
        previous_owner = owner
        new_codes[owner].extend(old_turn["audio_codes"])

    empty_code_turns = 0
    for text, codes in zip(new_texts, new_codes, strict=True):
        if codes:
            continue
        if any(char.isalnum() for char in text):
            raise ValueError(f"{row['id']}: spoken phrase turn without codec frames")
        empty_code_turns += 1
    old_frames = sum(len(turn["audio_codes"]) for turn in old_turns)
    new_frames = sum(len(codes) for codes in new_codes)
    if old_frames != new_frames:
        raise ValueError(f"{row['id']}: codec frame count changed")

    output = dict(row)
    output["conversations"] = [
        {"role": "assistant", "text": text, "audio_codes": codes}
        for text, codes in zip(new_texts, new_codes, strict=True)
    ]
    metadata = dict(output.get("metadata") or {})
    metadata.update(
        {
            "phrase_policy": "punctuation_boundary",
            "phrase_source_turns": len(old_turns),
            "phrase_output_turns": len(new_texts),
        }
    )
    output["metadata"] = metadata
    return output, punctuation_splits, empty_code_turns


def main() -> None:
    args = parse_args()
    if args.multiplier < 1:
        raise ValueError("--multiplier must be positive")
    step_seconds = args.chunk_seconds * args.multiplier
    max_hold_steps = max(1, round(args.phrase_max_hold_s / step_seconds))
    trajectories = read_trajectories(args.trajectory_tsv)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "rows": 0,
        "old_turns": 0,
        "new_turns": 0,
        "codec_frames": 0,
        "punctuation_only_boundary_splits": 0,
        "empty_code_punctuation_turns": 0,
        "multiplier": args.multiplier,
        "chunk_seconds": args.chunk_seconds,
        "phrase_max_hold_s": args.phrase_max_hold_s,
        "max_hold_steps": max_hold_steps,
        "phrase_min_chars": args.phrase_min_chars,
    }
    lengths: list[int] = []
    boundary_turns = 0
    with args.prepared_jsonl.open(encoding="utf-8") as source, args.output_jsonl.open(
        "w", encoding="utf-8"
    ) as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row["id"])
            trajectory_id = (
                row_id[len(args.id_prefix) :]
                if args.id_prefix and row_id.startswith(args.id_prefix)
                else row_id
            )
            if trajectory_id not in trajectories:
                raise KeyError(f"{row_id}: trajectory not found")
            new_texts = phrase_redistribute(
                trajectories[trajectory_id],
                args.multiplier,
                max_hold_steps,
                args.phrase_min_chars,
            )
            output, punctuation_splits, empty_code_turns = repack_row(row, new_texts)
            sink.write(json.dumps(output, ensure_ascii=False) + "\n")
            stats["rows"] += 1
            stats["old_turns"] += len(row["conversations"])
            stats["new_turns"] += len(output["conversations"])
            stats["codec_frames"] += sum(
                len(turn["audio_codes"]) for turn in output["conversations"]
            )
            stats["punctuation_only_boundary_splits"] += punctuation_splits
            stats["empty_code_punctuation_turns"] += empty_code_turns
            for text in new_texts:
                spoken_length = sum(char.isalnum() for char in text)
                lengths.append(spoken_length)
                if text.rstrip() and text.rstrip()[-1] in PHRASE_PUNCT:
                    boundary_turns += 1

    stats["median_spoken_chars"] = statistics.median(lengths)
    stats["turns_le_5_chars_ratio"] = sum(length <= 5 for length in lengths) / len(lengths)
    stats["phrase_boundary_ratio"] = boundary_turns / len(lengths)
    args.stats_json.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
