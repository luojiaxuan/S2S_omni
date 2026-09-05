#!/usr/bin/env python3
"""Phrase-gate an InfiniSST trajectory manifest: hold each release until it is a phrase, so
the thinker emits phrase-sized text deltas instead of the word fragments word alignment
produces.

A step releases when the held text reaches --release-chars spoken characters, or when it
ends at punctuation and already has --punct-min of them. Character count alone bounds the
hold, so there is no separate step limit; the punctuation minimum exists because without
it a buffer holding only "好。" is released as a two-character fragment, which is what this
stage removes (measured on 3000 rows: 3.2% of releases under four characters without it,
1.6% with it, at an unchanged median).

The rewrite only moves text LATER and never changes it: the concatenation of a row's
assistant turns is identical before and after, and the script refuses to write if it is
not. Audio paths are repointed at the local clip copy.

  phrase_gate_traj.py --in train_s_zh_origin.jsonl --out train_s_zh_phrase.jsonl \
      --release-chars 8 --punct-min 4 --audio-from <prefix> --audio-to <prefix>
"""
import argparse
import json
import os
import sys
from collections import Counter

PHRASE_PUNCT = "。，、？！；：,.?!;:"


def spoken_chars(text: str) -> int:
    return sum(1 for c in text if c.isalnum())


def gate(trajectory, release_chars: int, punct_min: int):
    """Return a same-length trajectory whose releases land on phrases."""
    out = [""] * len(trajectory)
    buffer = ""
    for index, text in enumerate(trajectory):
        buffer += text
        stripped = buffer.strip()
        if not stripped:
            continue
        held = spoken_chars(buffer)
        if held >= release_chars or (stripped[-1] in PHRASE_PUNCT and held >= punct_min):
            out[index] = buffer
            buffer = ""
    if buffer:
        out[-1] += buffer
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--release-chars", type=int, default=8)
    ap.add_argument("--punct-min", type=int, default=4)
    ap.add_argument("--audio-from", default="")
    ap.add_argument("--audio-to", default="")
    ap.add_argument("--check-audio", type=int, default=200)
    args = ap.parse_args()

    rows = releases_before = releases_after = 0
    len_before, len_after = Counter(), Counter()
    checked = missing = 0
    with open(args.src, encoding="utf-8") as src, open(args.dst, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["messages"]
            slots = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
            trajectory = [messages[i]["content"] for i in slots]
            gated = gate(trajectory, args.release_chars, args.punct_min)
            if "".join(gated) != "".join(trajectory):
                raise SystemExit(f"row {rows}: text changed — refusing to write")
            for slot, text in zip(slots, gated):
                messages[slot]["content"] = text
            releases_before += sum(1 for t in trajectory if t.strip())
            releases_after += sum(1 for t in gated if t.strip())
            for t in trajectory:
                if t.strip():
                    len_before[spoken_chars(t)] += 1
            for t in gated:
                if t.strip():
                    len_after[spoken_chars(t)] += 1
            if args.audio_from:
                row["audios"] = [p.replace(args.audio_from, args.audio_to) for p in row["audios"]]
                for p in row["audios"]:
                    if checked >= args.check_audio:
                        break
                    checked += 1
                    if not os.path.isfile(p):
                        missing += 1
                        if missing <= 3:
                            print(f"  MISSING AUDIO: {p}", file=sys.stderr)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1

    def median(counter):
        items = sorted(counter.elements())
        return items[len(items) // 2] if items else 0

    def mean(counter):
        total = sum(counter.values())
        return sum(k * v for k, v in counter.items()) / total if total else 0.0

    print(f"rows            : {rows}")
    print(f"releases        : {releases_before} -> {releases_after} "
          f"({releases_after / releases_before:.2f}x)")
    print(f"chars/release   : median {median(len_before)} -> {median(len_after)}, "
          f"mean {mean(len_before):.1f} -> {mean(len_after):.1f}")
    print(f"releases < 4 ch : {sum(v for k, v in len_before.items() if k < 4)} -> "
          f"{sum(v for k, v in len_after.items() if k < 4)}")
    print(f"audio checked   : {checked}, missing {missing}")
    if missing:
        raise SystemExit("rewritten audio paths do not exist")


if __name__ == "__main__":
    main()
