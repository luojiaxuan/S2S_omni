#!/usr/bin/env python3
"""Repetition-robustness augmentation for v2 multi-turn records.

For a sampled fraction of rows, duplicate a short span inside one middle
turn's audio codes (simulating the audio-loop degeneration observed at
inference) and mark that turn ``context_only`` so it carries no loss; the
following turns keep clean supervision, teaching the model to keep advancing
the text even when its own history contains a loop. Original rows are kept;
augmented rows are appended as extra copies with a ``_repaug`` id suffix.
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
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--min-span", type=int, default=8)
    parser.add_argument("--max-span", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=2, help="extra copies of the span")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = augmented = 0
    with out_path.open("w", encoding="utf-8") as out:
        for path in args.input_jsonl:
            for line in Path(path).open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                out.write(line + "\n")
                total += 1
                row = json.loads(line)
                turns = row.get("conversations") or []
                # need a corruptible middle turn with enough frames plus a successor
                candidates = [
                    i
                    for i, t in enumerate(turns[:-1])
                    if len(t.get("audio_codes") or []) >= args.min_span * 2
                ]
                if not candidates or rng.random() > args.fraction:
                    continue
                k = rng.choice(candidates)
                codes = [list(f) for f in turns[k]["audio_codes"]]
                span_len = rng.randint(args.min_span, min(args.max_span, len(codes) // 2))
                start = rng.randint(0, len(codes) - span_len)
                span = codes[start : start + span_len]
                injected = codes[: start + span_len] + span * args.repeats + codes[start + span_len :]
                new_turns = []
                for i, t in enumerate(turns):
                    t2 = dict(t)
                    if i == k:
                        t2["audio_codes"] = injected
                        t2["context_only"] = True
                    new_turns.append(t2)
                aug = dict(row)
                aug["id"] = str(row["id"]) + "_repaug"
                aug["conversations"] = new_turns
                meta = dict(aug.get("metadata") or {})
                meta["repetition_augmented_turn"] = k
                meta["repetition_span_frames"] = span_len
                aug["metadata"] = meta
                out.write(json.dumps(aug, ensure_ascii=False) + "\n")
                augmented += 1
    print(json.dumps({"rows": total, "augmented": augmented, "output": str(out_path)}))


if __name__ == "__main__":
    main()
