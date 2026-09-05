#!/usr/bin/env python3
"""Compare the two readings of "punctuation or character count"."""
import json, statistics, sys
sys.path.insert(0, "/home/jiaxuanluo")
from phrase_gate_traj_jaxan import spoken_chars, PHRASE_PUNCT

SRC = "/mnt/gemini/data/jiaxuanluo/manifests_rag/train_s_zh_origin.jsonl"
rows = []
with open(SRC, encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        if i >= 3000:
            break
        rows.append([m["content"] for m in json.loads(line)["messages"] if m["role"] == "assistant"])


def gate(trajectory, release_chars, punct_min):
    out = [""] * len(trajectory)
    buffer = ""
    for index, text in enumerate(trajectory):
        buffer += text
        stripped = buffer.strip()
        if not stripped:
            continue
        n = spoken_chars(buffer)
        if (stripped[-1] in PHRASE_PUNCT and n >= punct_min) or n >= release_chars:
            out[index] = buffer
            buffer = ""
    if buffer:
        out[-1] += buffer
    return out


base = [t for tr in rows for t in tr if t.strip()]
blens = sorted(spoken_chars(t) for t in base)
print(f"baseline: {len(base)} releases, median {statistics.median(blens)}, "
      f"<4ch {sum(1 for n in blens if n < 4)} ({100*sum(1 for n in blens if n<4)/len(blens):.0f}%)")
print()
print(f"{'punct_min':>9} {'release_chars':>13} {'releases':>9} {'median':>7} {'mean':>6} {'p90':>5} "
      f"{'<4ch':>6} {'<4ch%':>6}")
for punct_min in (0, 4, 6):
    for n in (8, 12, 16):
        outs = []
        for tr in rows:
            outs += [t for t in gate(tr, n, punct_min) if t.strip()]
        lens = sorted(spoken_chars(t) for t in outs)
        frag = sum(1 for x in lens if x < 4)
        print(f"{punct_min:>9} {n:>13} {len(outs):>9} {statistics.median(lens):>7.0f} "
              f"{statistics.mean(lens):>6.1f} {lens[int(0.9*len(lens))]:>5} {frag:>6} "
              f"{100*frag/len(lens):>5.1f}%")
