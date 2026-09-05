#!/usr/bin/env python3
"""Repoint a manifest's audio at this host's copy and verify EVERY file, not a sample.

The sampled check that shipped with phrase_gate_traj.py passed on 200 of 212k paths and
still let a run reach training with unreachable audio, where swift skipped the rows
silently. This walks the audio tree once into a set and checks all of them.
"""
import json, os, sys

SRC, DST, OLD, NEW = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

have = set()
for root, _dirs, files in os.walk(NEW):
    for f in files:
        if f.endswith(".wav"):
            have.add(os.path.join(root, f))
print(f"local wavs found: {len(have)}", flush=True)

rows = missing = repointed = 0
missing_examples = []
with open(SRC, encoding="utf-8") as src, open(DST, "w", encoding="utf-8") as dst:
    for line in src:
        if not line.strip():
            continue
        row = json.loads(line)
        paths = []
        for p in row["audios"]:
            q = p.replace(OLD, NEW)
            repointed += q != p
            if q not in have:
                missing += 1
                if len(missing_examples) < 3:
                    missing_examples.append(q)
            paths.append(q)
        row["audios"] = paths
        dst.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows += 1

print(f"rows {rows} | repointed {repointed} paths | missing {missing}")
for m in missing_examples:
    print("  missing:", m)
if missing:
    raise SystemExit("manifest references audio this host does not have")
print("ALL_AUDIO_PRESENT")
