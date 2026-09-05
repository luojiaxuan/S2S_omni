#!/usr/bin/env python3
"""Sweep the single knob of the phrase gate: how many characters a release may hold."""
import json, statistics, sys
sys.path.insert(0, "/home/jiaxuanluo")
from phrase_gate_traj_jaxan import gate, spoken_chars

SRC = "/mnt/gemini/data/jiaxuanluo/manifests_rag/train_s_zh_origin.jsonl"
rows = []
with open(SRC, encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        if i >= 3000:
            break
        rows.append([m["content"] for m in json.loads(line)["messages"] if m["role"] == "assistant"])

base = [t for tr in rows for t in tr if t.strip()]
blens = sorted(spoken_chars(t) for t in base)
print(f"baseline (word-aligned): {len(base)} releases, median {statistics.median(blens)}, "
      f"mean {statistics.mean(blens):.1f}, p90 {blens[int(0.9*len(blens))]}, "
      f"<4ch {sum(1 for n in blens if n < 4)}")
print()
print(f"{'release_chars':>13} {'releases':>9} {'ratio':>6} {'median':>7} {'mean':>6} {'p90':>5} {'<4ch':>6} {'>=24ch':>7}")
for n in (6, 8, 10, 12, 14, 16, 20, 24):
    outs = []
    for tr in rows:
        outs += [t for t in gate(tr, n) if t.strip()]
    lens = sorted(spoken_chars(t) for t in outs)
    print(f"{n:>13} {len(outs):>9} {len(outs)/len(base):>6.2f} {statistics.median(lens):>7.0f} "
          f"{statistics.mean(lens):>6.1f} {lens[int(0.9*len(lens))]:>5} "
          f"{sum(1 for x in lens if x < 4):>6} {sum(1 for x in lens if x >= 24):>7}")
