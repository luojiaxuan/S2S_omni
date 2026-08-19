#!/usr/bin/env python3
"""Convert a trajectory-aligned TSV (lab GigaSpeech/FLORAS pipeline) into MOSS
row requests consumable by generate_moss_realtime_long_targets.py.

台账 4.-19。输入是 lab 数据管线 stage-7 的产物：每行一个英文 utterance，
`trajectory` 列是按 0.96s chunk 对齐的中文增量（空串 = 等待）。已验证
6385/6387 行的 trajectory 拼接与 tgt_text 完全一致，因此可以整段合成后按
turn 边界强制对齐切片——与既有管线的约定完全相同：

- 非空增量即 turn；纯标点 turn 无法强制对齐，并入前一个 turn（行首标点并入
  后一个），复用 build_moss_v2_row_requests 的约定；
- 超过 --max-chars-per-call 的行按句末优先切成合成分组；
- turn 中位仅 4 字（0.96s chunk 比 InfiniSST 的 1.92s 细一倍）——短 turn
  正是级联的历史弱点（滑窗吞短句，台账 4.-6），这批数据是有针对性的补充。

usage:
  build_moss_rows_from_trajectory.py --tsv train_..._align.tsv \
      --output-jsonl rows_traj.jsonl [--id-prefix traj] [--max-rows 0]
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import unicodedata
from pathlib import Path

SENTENCE_FINAL = "。！？!?…"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--id-prefix", default="traj")
    parser.add_argument("--max-chars-per-call", type=int, default=300)
    parser.add_argument("--max-duration-s", type=float, default=120.0,
                        help="drop utterances longer than this (runaway 训练风险，"
                             "沿用 30s 规则的精神但放宽——本数据集中位 38.4s)")
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def spoken_chars(text: str) -> int:
    return sum(1 for ch in text if unicodedata.category(ch)[0] in ("L", "N"))


def main() -> None:
    args = parse_args()
    csv.field_size_limit(10**8)
    stats = {"rows_in": 0, "dirty": 0, "too_long": 0, "no_turns": 0,
             "rows_out": 0, "turns_out": 0, "punct_merged": 0}

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.tsv.open(newline="") as handle, args.output_jsonl.open("w") as out:
        for row in csv.DictReader(handle, delimiter="\t"):
            stats["rows_in"] += 1
            raw = (row.get("trajectory") or "").lstrip()
            if not raw.startswith("["):
                stats["dirty"] += 1
                continue
            try:
                traj = ast.literal_eval(raw)
                assert isinstance(traj, list)
            except (ValueError, SyntaxError, AssertionError):
                stats["dirty"] += 1
                continue
            if int(row["n_frames"]) / 16000 > args.max_duration_s:
                stats["too_long"] += 1
                continue

            # 非空增量 -> turn；纯标点 turn 并入邻居（同 v2 约定）
            deltas = [t for t in traj if t.strip()]
            merged: list[str] = []
            for d in deltas:
                if merged and spoken_chars(d) == 0:
                    merged[-1] += d
                    stats["punct_merged"] += 1
                elif not merged and spoken_chars(d) == 0 and len(deltas) > 1:
                    # 行首纯标点：挂到占位符，与下一个增量合并
                    merged.append(d)
                    stats["punct_merged"] += 1
                else:
                    if merged and spoken_chars(merged[-1]) == 0:
                        merged[-1] += d
                    else:
                        merged.append(d)
            merged = [m for m in merged if spoken_chars(m) > 0]
            if not merged:
                stats["no_turns"] += 1
                continue

            row_id = f"{args.id_prefix}_{row['id']}"
            segments = [{"id": f"{row_id}_t{i:03d}", "text": t} for i, t in enumerate(merged)]

            # 句末优先的合成分组（同 build_moss_v2_row_requests）
            groups: list[list[int]] = []
            current: list[int] = []
            current_chars = 0
            for idx, seg in enumerate(segments):
                seg_chars = len(seg["text"])
                if current and current_chars + seg_chars > args.max_chars_per_call:
                    groups.append(current)
                    current, current_chars = [], 0
                current.append(idx)
                current_chars += seg_chars
                if (current_chars > args.max_chars_per_call * 0.7
                        and seg["text"].rstrip()
                        and seg["text"].rstrip()[-1] in SENTENCE_FINAL):
                    groups.append(current)
                    current, current_chars = [], 0
            if current:
                groups.append(current)

            record = {
                "row_id": row_id,
                "split": "train",
                "segments": segments,
                "groups": groups,
                "full_text": "".join(s["text"] for s in segments),
                "spoken_chars": sum(spoken_chars(s["text"]) for s in segments),
                "source": "lab_trajectory_tsv",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["rows_out"] += 1
            stats["turns_out"] += len(segments)
            if args.max_rows and stats["rows_out"] >= args.max_rows:
                break

    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
