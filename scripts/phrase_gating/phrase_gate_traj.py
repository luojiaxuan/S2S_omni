#!/usr/bin/env python3
"""Phrase-gate an InfiniSST trajectory manifest: hold every release until it completes a
phrase, so the thinker emits phrase-sized text deltas instead of word fragments.

The rewrite moves text LATER in the trajectory and never changes it: the concatenation of
a row's assistant turns is identical before and after, and the script fails if it is not.
Audio paths are repointed at the local clip copy.

  phrase_gate_traj.py --in train_s_zh_origin.jsonl --out train_s_zh_phrase.jsonl \
      --min-chars 6 --max-hold 8 --audio-from <prefix> --audio-to <prefix>
"""
import argparse
import json
import os
import sys
from collections import Counter

PHRASE_PUNCT = "。，、？！；：,.?!;:"


def spoken_chars(text: str) -> int:
    return sum(1 for c in text if c.isalnum())


def gate(trajectory, min_chars: int, max_hold: int):
    """Return a same-length trajectory whose releases land on phrase boundaries."""
    out = [""] * len(trajectory)
    buffer, held = "", 0
    for index, text in enumerate(trajectory):
        buffer += text
        stripped = buffer.strip()
        if not stripped:
            continue
        held += 1
        at_boundary = stripped[-1] in PHRASE_PUNCT and spoken_chars(buffer) >= min_chars
        if at_boundary or held >= max_hold or index == len(trajectory) - 1:
            out[index] = buffer
            buffer, held = "", 0
    if buffer:
        out[-1] += buffer
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--min-chars", type=int, default=6)
    ap.add_argument("--max-hold", type=int, default=8)
    ap.add_argument("--audio-from", default="")
    ap.add_argument("--audio-to", default="")
    ap.add_argument("--check-audio", type=int, default=200,
                    help="verify this many rewritten audio paths exist (0 = skip)")
    args = ap.parse_args()

    rows = releases_before = releases_after = 0
    len_before = Counter()
    len_after = Counter()
    checked = missing = 0
    with open(args.src, encoding="utf-8") as src, open(args.dst, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["messages"]
            slots = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
            trajectory = [messages[i]["content"] for i in slots]
            gated = gate(trajectory, args.min_chars, args.max_hold)
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
                if checked < args.check_audio:
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

    print(f"rows            : {rows}")
    print(f"releases        : {releases_before} -> {releases_after} "
          f"({releases_after / releases_before:.2f}x)")
    print(f"chars/release   : median {median(len_before)} -> {median(len_after)}, "
          f"mean {sum(k*v for k,v in len_before.items())/max(sum(len_before.values()),1):.1f} -> "
          f"{sum(k*v for k,v in len_after.items())/max(sum(len_after.values()),1):.1f}")
    print(f"releases < 4 ch : {sum(v for k,v in len_before.items() if k<4)} -> "
          f"{sum(v for k,v in len_after.items() if k<4)}")
    print(f"audio checked   : {checked}, missing {missing}")
    if missing:
        raise SystemExit("rewritten audio paths do not exist")


if __name__ == "__main__":
    main()
