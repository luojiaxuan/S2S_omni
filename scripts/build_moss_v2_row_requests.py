#!/usr/bin/env python3
"""Build row-level MOSS v2 generation requests from segment manifests.

v2 synthesizes each InfiniSST row's full Chinese target text as one long
utterance (fixed voice), then forced-aligns segment boundaries and slices
codec codes. This script groups the v1 segment manifest into rows.

# note (luojiaxuan): punct-only segments cannot be force-aligned (no spoken
# chars), so their text is merged into the previous segment (or the next one
# when the row starts with punctuation). Rows whose full text exceeds
# --max-chars-per-call are split into sentence-preferring groups at segment
# boundaries; groups are synthesized separately and the audio is concatenated.
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path


def has_spoken_content(text: str) -> bool:
    return any(unicodedata.category(ch)[0] in {"L", "N"} for ch in text)


def spoken_chars(text: str) -> int:
    return sum(1 for ch in text if unicodedata.category(ch)[0] in {"L", "N"})


SENTENCE_FINAL = "。！？!?；;"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-chars-per-call", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: dict[str, list[dict]] = defaultdict(list)
    split = None
    with Path(args.manifest_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            row_id, turn = req["id"].rsplit("_t", 1)
            split = req.get("split") or split
            rows[row_id].append(
                {"id": req["id"], "turn": int(turn), "text": str(req["target_text"])}
            )

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = n_segments = n_merged = n_groups = 0
    with out_path.open("w", encoding="utf-8") as out:
        for row_id in sorted(rows):
            segments = sorted(rows[row_id], key=lambda s: s["turn"])
            # merge punct-only segments into a spoken neighbour
            merged: list[dict] = []
            pending_prefix = ""
            for seg in segments:
                if has_spoken_content(seg["text"]):
                    if pending_prefix:
                        seg = {**seg, "text": pending_prefix + seg["text"]}
                        pending_prefix = ""
                    merged.append(dict(seg))
                elif merged:
                    merged[-1]["text"] += seg["text"]
                    n_merged += 1
                else:
                    pending_prefix += seg["text"]
                    n_merged += 1
            if pending_prefix and merged:
                merged[0]["text"] = pending_prefix + merged[0]["text"]
            if not merged:
                continue

            # group segments so each generation call stays under the char cap,
            # preferring split points after sentence-final punctuation
            groups: list[list[int]] = []
            current: list[int] = []
            current_chars = 0
            for idx, seg in enumerate(merged):
                seg_chars = len(seg["text"])
                if current and current_chars + seg_chars > args.max_chars_per_call:
                    groups.append(current)
                    current = []
                    current_chars = 0
                current.append(idx)
                current_chars += seg_chars
                if (
                    current_chars > args.max_chars_per_call * 0.7
                    and seg["text"].rstrip()
                    and seg["text"].rstrip()[-1] in SENTENCE_FINAL
                ):
                    groups.append(current)
                    current = []
                    current_chars = 0
            if current:
                groups.append(current)

            record = {
                "row_id": row_id,
                "split": split,
                "segments": [{"id": s["id"], "text": s["text"]} for s in merged],
                "groups": groups,
                "full_text": "".join(s["text"] for s in merged),
                "spoken_chars": sum(spoken_chars(s["text"]) for s in merged),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_rows += 1
            n_segments += len(merged)
            n_groups += len(groups)

    print(
        json.dumps(
            {
                "manifest": args.manifest_jsonl,
                "output": str(out_path),
                "rows": n_rows,
                "segments_after_merge": n_segments,
                "punct_only_merged": n_merged,
                "generation_groups": n_groups,
            }
        )
    )


if __name__ == "__main__":
    main()
