#!/usr/bin/env python3
"""Sweep the phrase gate's two knobs and report what each setting does to the releases."""
import json, statistics, sys
sys.path.insert(0, "/home/jiaxuanluo")
from phrase_gate_traj_jaxan import gate, spoken_chars

SRC = "/mnt/gemini/data/jiaxuanluo/manifests_rag/train_s_zh_origin.jsonl"
rows = []
with open(SRC, encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        if i >= 3000:
            break
        r = json.loads(line)
        rows.append([m["content"] for m in r["messages"] if m["role"] == "assistant"])

base = [t for tr in rows for t in tr if t.strip()]
print(f"baseline (word-aligned): {len(base)} releases, median {statistics.median(spoken_chars(t) for t in base)}"
      f", mean {statistics.mean(spoken_chars(t) for t in base):.1f}, <4ch {sum(1 for t in base if spoken_chars(t)<4)}")
print()
print(f"{'min_chars':>9} {'max_hold':>9} {'releases':>9} {'ratio':>6} {'median':>7} {'mean':>6} {'p90':>5} {'<4ch':>6}")
for min_chars in (2, 3, 4, 6):
    for max_hold in (2, 3, 4, 6, 8):
        outs = []
        for tr in rows:
            outs += [t for t in gate(tr, min_chars, max_hold) if t.strip()]
        lens = sorted(spoken_chars(t) for t in outs)
        print(f"{min_chars:>9} {max_hold:>9} {len(outs):>9} {len(outs)/len(base):>6.2f} "
              f"{statistics.median(lens):>7.0f} {statistics.mean(lens):>6.1f} "
              f"{lens[int(0.9*len(lens))]:>5} {sum(1 for n in lens if n<4):>6}")
