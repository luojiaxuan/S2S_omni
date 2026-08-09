#!/usr/bin/env python3
"""Build the v4 training set: self-generated history (turn-level scheduled sampling).

与 v3 的唯一区别是**污染来源**：v3 用我手写的三种坏历史（轮内跳针 / 跨轮
carry / 整轮 replay），v4 用 ``gen_moss_self_history.py`` 跑出来的模型自身
闭环输出。其余结构保持一致，这样 v3 vs v4 是一次干净的 A/B：
"手写错误分布" vs "模型真实错误分布"。

替换粒度是 turn 级而非帧级：自生成 turn 与真值 turn 帧数不同，无法逐帧对齐；
turn 边界是天然的对齐点。被替换的 turn 标 ``context_only``（进上下文、不进
loss，靠已打的 dataset.py 补丁生效），未被替换的 turn 仍用干净真值做监督。
因此格式与 v3 完全兼容，prepare_data.py / sft.py 无需任何改动。

漂移护栏：自生成 turn 若时长相对真值偏离过大（或生成时被判 runaway），回退
用真值。否则后续真值 turn 会接在一个时长/韵律都跑偏的上下文之后，等于教模型
不自然的衔接——这是 scheduled sampling 在语音上最现实的坑。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-jsonl", nargs="+", required=True,
                        help="v3-style clean records (originals + long sessions, no context_only)")
    parser.add_argument("--self-history-jsonl", nargs="+", required=True,
                        help="output of gen_moss_self_history.py")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--fraction", type=float, default=0.5,
                        help="fraction of eligible rows that get a self-history copy")
    parser.add_argument("--replace-prob", type=float, default=0.5,
                        help="per-turn probability of swapping in self-generated codes")
    parser.add_argument("--max-drift", type=float, default=2.0,
                        help="skip swap when self_frames/gt_frames is outside [1/max_drift, max_drift]")
    # note (luojiaxuan): DAgger 多轮聚合时要能只产副本、不重复输出干净行
    # （否则第二轮会把 14,485 行干净数据再写一遍）。原来写成 store_true +
    # default=True，标志只能置真、关不掉，是个 bug。
    parser.add_argument("--keep-clean-copies", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--id-suffix", default="_v4self",
                        help="suffix for generated copies; use a distinct one per DAgger round "
                             "so ids from different rounds do not collide")
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def load_self_history(paths: list[str]) -> dict[str, list[dict]]:
    table: dict[str, list[dict]] = {}
    for path in paths:
        with open(path) as handle:
            for line in handle:
                row = json.loads(line)
                table[row["id"]] = row["turns"]
    return table


def build_self_copy(rng: random.Random, row: dict, self_turns: list[dict],
                    replace_prob: float, max_drift: float,
                    id_suffix: str = "_v4self") -> dict | None:
    turns = row["conversations"]
    if len(self_turns) < len(turns):
        return None
    new_turns = []
    swapped = 0
    # note (luojiaxuan): 最后一个 turn 永远保持干净真值监督——它是这一行里
    # 最贴近推理时"给定脏历史产出干净下一轮"的那个位置。
    for k, turn in enumerate(turns):
        is_last = k == len(turns) - 1
        cand = self_turns[k]
        gt_frames = len(turn.get("audio_codes") or [])
        self_frames = int(cand.get("self_frames") or 0)
        drift_ok = (
            gt_frames > 0 and self_frames > 0
            and (1.0 / max_drift) <= (self_frames / gt_frames) <= max_drift
            and not cand.get("runaway")
        )
        if not is_last and drift_ok and rng.random() < replace_prob:
            new_turn = dict(turn)
            new_turn["audio_codes"] = cand["self_audio_codes"]
            new_turn["context_only"] = True
            new_turns.append(new_turn)
            swapped += 1
        else:
            new_turns.append(dict(turn))
    if swapped == 0:
        return None
    out = dict(row)
    out["id"] = f"{row['id']}{id_suffix}"
    out["conversations"] = new_turns
    meta = dict(row.get("metadata") or {})
    meta["v4_self_history_turns"] = swapped
    meta["v4_total_turns"] = len(new_turns)
    out["metadata"] = meta
    return out


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    self_table = load_self_history(args.self_history_jsonl)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"clean_rows": 0, "self_rows": 0, "swapped_turns": 0,
             "eligible_rows": 0, "skipped_no_self": 0, "skipped_no_swap": 0}

    with out_path.open("w") as sink:
        for path in args.clean_jsonl:
            with open(path) as handle:
                for line in handle:
                    row = json.loads(line)
                    if args.keep_clean_copies:
                        sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stats["clean_rows"] += 1
                    self_turns = self_table.get(row["id"])
                    if self_turns is None:
                        stats["skipped_no_self"] += 1
                        continue
                    stats["eligible_rows"] += 1
                    if rng.random() >= args.fraction:
                        continue
                    copy = build_self_copy(rng, row, self_turns,
                                           args.replace_prob, args.max_drift,
                                           args.id_suffix)
                    if copy is None:
                        stats["skipped_no_swap"] += 1
                        continue
                    sink.write(json.dumps(copy, ensure_ascii=False) + "\n")
                    stats["self_rows"] += 1
                    stats["swapped_turns"] += copy["metadata"]["v4_self_history_turns"]

    stats["total_rows"] = stats["clean_rows"] + stats["self_rows"]
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
