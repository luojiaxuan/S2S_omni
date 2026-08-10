#!/usr/bin/env python3
"""Build the v6 training set: rows that start mid-passage (no silence anchor).

诊断链见实验台账 4.-6 / 4.-7 / 4.-8。要点：

- 滑窗相对 reset 掉约 3.4 BLEU，损失集中在**短 turn 被吞进上一句**；
- 推理侧重生成修不好（系统性行为，不是采样偶发）；
- "短 turn 上下文罕见"被分布统计否定（训练里反而更多）；
- 真正的分布缺口是**静音锚点**：`align_slice` 让每行都从整段合成的第 0 帧
  起（`frame_cuts[0] = 0`），所以**每个训练上下文都含一个"从静音开始"的
  turn**；而滑窗推理的窗口里几乎永远不含。这也解释了 4.1 的核心发现——
  同样的平均历史长度下，会周期性清零的 reset 比从不清零的滑窗高 8 分。

做法：对一部分行，丢掉开头 k 个 turn，让首 turn 的音频从段落中间开始。
不需要重新合成，纯粹是换切片起点，因此几乎零成本。监督保持干净
（不打 ``context_only``），只改变上下文的起始形态。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-jsonl", nargs="+", required=True,
                        help="clean multi-turn records to derive mid-passage copies from")
    parser.add_argument("--base-jsonl", nargs="*", default=[],
                        help="dataset to prepend unchanged (e.g. train_v5.jsonl); "
                             "omit to emit only the mid-passage copies")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--fraction", type=float, default=0.5,
                        help="fraction of eligible clean rows that get a mid-passage copy")
    parser.add_argument("--min-remaining-turns", type=int, default=4,
                        help="a mid-passage copy must keep at least this many turns")
    parser.add_argument("--max-drop-frac", type=float, default=0.5,
                        help="drop at most this fraction of a row's turns from the front")
    parser.add_argument("--id-suffix", default="_v6mid")
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"base_rows": 0, "eligible": 0, "mid_rows": 0,
             "skipped_too_short": 0, "dropped_turns_total": 0}

    with out_path.open("w") as sink:
        for path in args.base_jsonl:
            with open(path) as handle:
                for line in handle:
                    sink.write(line)
                    stats["base_rows"] += 1

        for path in args.clean_jsonl:
            with open(path) as handle:
                for line in handle:
                    row = json.loads(line)
                    turns = row["conversations"]
                    max_drop = int(len(turns) * args.max_drop_frac)
                    if len(turns) - 1 < args.min_remaining_turns or max_drop < 1:
                        stats["skipped_too_short"] += 1
                        continue
                    stats["eligible"] += 1
                    if rng.random() >= args.fraction:
                        continue
                    hi = min(max_drop, len(turns) - args.min_remaining_turns)
                    if hi < 1:
                        stats["skipped_too_short"] += 1
                        continue
                    k = rng.randint(1, hi)
                    out = dict(row)
                    out["id"] = f"{row['id']}{args.id_suffix}{k}"
                    out["conversations"] = [dict(t) for t in turns[k:]]
                    meta = dict(row.get("metadata") or {})
                    # note (luojiaxuan): 记下丢了几个 turn，方便事后核对
                    # "首 turn 是否真的从段落中间开始"这一构造意图。
                    meta["v6_dropped_leading_turns"] = k
                    meta["v6_remaining_turns"] = len(out["conversations"])
                    out["metadata"] = meta
                    sink.write(json.dumps(out, ensure_ascii=False) + "\n")
                    stats["mid_rows"] += 1
                    stats["dropped_turns_total"] += k

    stats["total_rows"] = stats["base_rows"] + stats["mid_rows"]
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
