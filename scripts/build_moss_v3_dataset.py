#!/usr/bin/env python3
"""Build the v3 training set: long sessions + strong repetition corruption.

Two transforms over v2 prepared multi-turn records:
1. Long sessions — concatenate consecutive rows (row-id order approximates
   stream order; all rows share the fixed voice ensemble) into sessions of
   --min-turns..--max-turns turns, matching the sliding-window shape the
   model sees at full-talk inference. Originals are kept alongside.
2. Strong corruption — for --fraction of all rows (original + long), corrupt
   1..3 turns as ``context_only`` (no loss) with one of three loop patterns:
   intra-turn span duplication, cross-turn span carry-over (previous turn's
   tail spliced into the current turn), or full previous-turn replay.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", nargs="+", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--min-turns", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=35)
    parser.add_argument("--long-session-copies", type=float, default=1.0,
                        help="long-session rows to emit, as a fraction of the source stream coverage")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--min-span", type=int, default=8)
    parser.add_argument("--max-span", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def corrupt_turn(rng: random.Random, turns: list[dict], k: int, min_span: int, max_span: int) -> dict:
    turn = dict(turns[k])
    codes = [list(f) for f in turn.get("audio_codes") or []]
    prev_codes = [list(f) for f in (turns[k - 1].get("audio_codes") or [])] if k > 0 else []
    mode = rng.choice(["intra", "carry", "replay"] if prev_codes else ["intra"])
    if mode == "intra" and len(codes) >= min_span * 2:
        span_len = rng.randint(min_span, min(max_span, len(codes) // 2))
        start = rng.randint(0, len(codes) - span_len)
        span = codes[start : start + span_len]
        codes = codes[: start + span_len] + span * rng.randint(1, 3) + codes[start + span_len :]
    elif mode == "carry" and len(prev_codes) >= min_span:
        span_len = rng.randint(min_span, min(max_span, len(prev_codes)))
        span = prev_codes[-span_len:]
        codes = span * rng.randint(1, 2) + codes
    elif mode == "replay" and prev_codes:
        codes = prev_codes + codes[: max(0, len(codes) // 4)]
    else:
        return turns[k]
    turn["audio_codes"] = codes
    turn["context_only"] = True
    return turn


def apply_corruption(rng: random.Random, row: dict, min_span: int, max_span: int) -> dict | None:
    turns = row.get("conversations") or []
    # keep the final turn clean so every record ends with supervised advance
    candidates = [i for i in range(len(turns) - 1) if len(turns[i].get("audio_codes") or []) >= min_span]
    if not candidates:
        return None
    picks = rng.sample(candidates, k=min(len(candidates), rng.randint(1, 3)))
    new_turns = list(turns)
    for k in sorted(picks):
        new_turns[k] = corrupt_turn(rng, new_turns, k, min_span, max_span)
    if not any(t.get("context_only") for t in new_turns):
        return None
    aug = dict(row)
    aug["id"] = str(row["id"]) + "_v3corrupt"
    aug["conversations"] = new_turns
    meta = dict(aug.get("metadata") or {})
    meta["v3_corrupted_turns"] = sorted(picks)
    aug["metadata"] = meta
    return aug


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows: list[dict] = []
    for path in args.input_jsonl:
        for line in Path(path).open(encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: str(r["id"]))

    long_rows: list[dict] = []
    i = 0
    while i < len(rows):
        target = rng.randint(args.min_turns, args.max_turns)
        turns: list[dict] = []
        member_ids: list[str] = []
        j = i
        while j < len(rows) and len(turns) < target:
            turns.extend(rows[j]["conversations"])
            member_ids.append(str(rows[j]["id"]))
            j += 1
        if len(turns) >= args.min_turns:
            base = dict(rows[i])
            base["id"] = f"v3long_{rows[i]['id']}_{len(member_ids)}rows"
            base["conversations"] = turns
            meta = dict(base.get("metadata") or {})
            meta["v3_long_session_members"] = member_ids
            base["metadata"] = meta
            long_rows.append(base)
        i = j

    combined = rows + long_rows
    corrupted: list[dict] = []
    for row in combined:
        if rng.random() < args.fraction:
            aug = apply_corruption(rng, row, args.min_span, args.max_span)
            if aug is not None:
                corrupted.append(aug)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for row in combined + corrupted:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "originals": len(rows),
                "long_sessions": len(long_rows),
                "corrupted": len(corrupted),
                "total": len(combined) + len(corrupted),
                "output": str(out_path),
            }
        )
    )


if __name__ == "__main__":
    main()
