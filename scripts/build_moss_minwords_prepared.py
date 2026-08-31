#!/usr/bin/env python3
"""v9 配对数据：按 min-words 规则重排 TTS 训练 turn，与 InfiniSST v9 collator 同语义。

与 build_moss_phrase_prepared.py 的差别：
- 释放规则换成 min_words（无标点判定、无 max_hold）；
- multiplier 不再全局固定，而是按行 id 稳定哈希分配到 {1,2,3,4}，与 InfiniSST
  训练时的随机采样在分布上匹配，且可复现。
repack 复用同一套守恒断言：拼接文本逐字不变、codec 帧数不变、
新边界只落在旧 delta 边界上。
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_moss_phrase_prepared import read_trajectories, repack_row, PHRASE_PUNCT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory-tsv", required=True, type=Path)
    p.add_argument("--prepared-jsonl", required=True, type=Path)
    p.add_argument("--output-jsonl", required=True, type=Path)
    p.add_argument("--stats-json", required=True, type=Path)
    p.add_argument("--id-prefix", default="traj_")
    p.add_argument("--multipliers", default="1,2,3,4")
    p.add_argument("--chunk-seconds", type=float, default=0.96)
    p.add_argument("--min-words", type=int, default=2)
    return p.parse_args()


def pick_multiplier(row_id: str, choices: list[int]) -> int:
    h = int.from_bytes(hashlib.sha1(row_id.encode()).digest()[:4], "big")
    return choices[h % len(choices)]


def minwords_redistribute(trajectory: list[str], multiplier: int, min_words: int) -> list[str]:
    grouped = ["".join(trajectory[i:i + multiplier])
               for i in range(0, len(trajectory), multiplier)]
    outputs: list[str] = []
    buf = ""
    for i, text in enumerate(grouped):
        buf += text
        if not buf.strip():
            continue
        if sum(c.isalnum() for c in buf) >= min_words or i == len(grouped) - 1:
            outputs.append(buf)
            buf = ""
    if buf:
        if outputs:
            outputs[-1] += buf
        else:
            outputs.append(buf)
    return outputs


def main() -> None:
    args = parse_args()
    choices = [int(x) for x in args.multipliers.split(",")]
    trajectories = read_trajectories(args.trajectory_tsv)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {"rows": 0, "old_turns": 0, "new_turns": 0,
                             "codec_frames": 0, "min_words": args.min_words,
                             "multipliers": choices,
                             "per_multiplier_rows": {str(c): 0 for c in choices}}
    lengths, boundary = [], 0
    with args.prepared_jsonl.open(encoding="utf-8") as src, \
         args.output_jsonl.open("w", encoding="utf-8") as sink:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row["id"])
            tid = row_id[len(args.id_prefix):] if row_id.startswith(args.id_prefix) else row_id
            if tid not in trajectories:
                raise KeyError(f"{row_id}: trajectory not found")
            mult = pick_multiplier(row_id, choices)
            new_texts = minwords_redistribute(trajectories[tid], mult, args.min_words)
            output, _, _ = repack_row(row, new_texts)
            output["metadata"]["phrase_policy"] = "min_words_v9"
            output["metadata"]["multiplier"] = mult
            sink.write(json.dumps(output, ensure_ascii=False) + "\n")
            stats["rows"] += 1
            stats["per_multiplier_rows"][str(mult)] += 1
            stats["old_turns"] += len(row["conversations"])
            stats["new_turns"] += len(output["conversations"])
            stats["codec_frames"] += sum(len(t["audio_codes"]) for t in output["conversations"])
            for text in new_texts:
                n = sum(c.isalnum() for c in text)
                lengths.append(n)
                if text.rstrip() and text.rstrip()[-1] in PHRASE_PUNCT:
                    boundary += 1
    stats["median_spoken_chars"] = statistics.median(lengths)
    stats["turns_le_5_chars_ratio"] = round(sum(x <= 5 for x in lengths) / len(lengths), 4)
    stats["phrase_boundary_ratio"] = round(boundary / len(lengths), 4)
    args.stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
