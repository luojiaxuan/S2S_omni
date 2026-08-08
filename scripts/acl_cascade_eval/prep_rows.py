#!/usr/bin/env python3
# note (luojiaxuan): 从 v3 数据集里切出干净行（不含任何 context_only 的 turn）
# 作为 v4 的基底，保证 v4 与 v3 共享完全相同的原始行和长会话，A/B 只差污染来源。
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

clean = out_dir / "clean_rows.jsonl"
originals = out_dir / "clean_originals.jsonl"
longsess = out_dir / "clean_longsess.jsonl"

n_clean = n_orig = n_long = n_corrupt = 0
with src.open() as handle, clean.open("w") as fc, originals.open("w") as fo, longsess.open("w") as fl:
    for line in handle:
        row = json.loads(line)
        if any(t.get("context_only") for t in row["conversations"]):
            n_corrupt += 1
            continue
        fc.write(line)
        n_clean += 1
        # note (luojiaxuan): v3 builder 给长会话行的 id 带 _v3long 后缀
        if "_v3long" in row["id"]:
            fl.write(line)
            n_long += 1
        else:
            fo.write(line)
            n_orig += 1

turns_orig = sum(1 for _ in originals.open())
print(json.dumps({
    "clean_rows": n_clean, "originals": n_orig, "long_sessions": n_long,
    "dropped_corrupted": n_corrupt,
}, indent=1))
