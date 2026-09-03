#!/usr/bin/env python3
"""在每个 chunk 大小上扫 min_chars × max_hold，找「落标点比例」最高的配置。

chunk 大小（multiplier × 0.96s）是要报告的延迟横轴，不参与调优；每个 chunk
大小上独立选另外两个参数。max_hold 直接以**步数**给定——原实现用秒换算，
chunk 变大时步数塌到 1-2，等于强制切断。

配对消融（punct_gate_matched_ablation_20260830）显示：同字数下以标点结尾的
write，TTS 字错率与超额生成率都约为半。故目标函数是「落标点比例」，
在延迟约束下最大化。
usage: sweep_write_policy.py --trajectory-tsv T --ids-jsonl C
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import statistics
from pathlib import Path

PHRASE_PUNCT = "。！？!?…，,、；;：:"
CHUNK_S = 0.96


def alnum_n(s: str) -> int:
    return sum(c.isalnum() for c in s)


def simulate(grouped: list[str], max_hold: int, min_chars: int):
    """返回 [(write_text, 首块等待步数, 各字符等待步数之和, 字符数)]"""
    out, buf, held, waits = [], "", 0, []
    for idx, text in enumerate(grouped):
        buf += text
        if text:
            waits.append((len(text), idx))
        s = buf.strip()
        if not s:
            held = 0
            continue
        held += 1
        at_boundary = s[-1] in PHRASE_PUNCT and alnum_n(buf) >= min_chars
        if at_boundary or held >= max_hold or idx == len(grouped) - 1:
            first = waits[0][1] if waits else idx
            char_wait = sum(n * (idx - c) for n, c in waits)
            out.append((buf, idx - first, char_wait, sum(n for n, _ in waits)))
            buf, held, waits = "", 0, []
    if buf.strip():
        out.append((buf, 0, 0, len(buf)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory-tsv", required=True, type=Path)
    ap.add_argument("--ids-jsonl", required=True, type=Path)
    ap.add_argument("--id-prefix", default="traj_")
    ap.add_argument("--out-json", type=Path)
    args = ap.parse_args()

    keep = set()
    for line in args.ids_jsonl.open(encoding="utf-8"):
        m = re.match(r'\{"id":\s*"([^"]+)"', line)
        if m:
            i = m.group(1)
            keep.add(i[len(args.id_prefix):] if i.startswith(args.id_prefix) else i)

    csv.field_size_limit(10 ** 8)
    trajs = []
    with args.trajectory_tsv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if str(row["id"]) not in keep:
                continue
            raw = (row.get("trajectory") or "").lstrip()
            if not raw.startswith("["):
                continue
            try:
                t = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                continue
            if isinstance(t, list) and all(isinstance(x, str) for x in t):
                trajs.append(t)

    results = []
    for mult in (1, 2, 3, 4):
        step_s = CHUNK_S * mult
        grouped_all = [["".join(t[i:i + mult]) for i in range(0, len(t), mult)]
                       for t in trajs]
        for min_chars in (2, 4, 6, 8, 12):
            for max_hold in (2, 3, 4, 6, 8, 12):
                n_w = n_punct = 0
                lens, first_waits, char_wait_sum, char_total = [], [], 0, 0
                for g in grouped_all:
                    for text, fw, cw, nc in simulate(g, max_hold, min_chars):
                        s = text.strip()
                        if not s:
                            continue
                        n_w += 1
                        n_punct += s[-1] in PHRASE_PUNCT
                        lens.append(alnum_n(s))
                        first_waits.append(fw * step_s)
                        char_wait_sum += cw * step_s
                        char_total += nc
                first_waits.sort()
                results.append({
                    "chunk_s": round(step_s, 2), "multiplier": mult,
                    "min_chars": min_chars, "max_hold_steps": max_hold,
                    "max_hold_s": round(max_hold * step_s, 2),
                    "writes": n_w,
                    "pct_ends_punct": round(n_punct / n_w * 100, 1),
                    "median_chars": statistics.median(lens),
                    "pct_len_ge16": round(sum(x >= 16 for x in lens) / len(lens) * 100, 1),
                    "char_delay_mean_s": round(char_wait_sum / char_total, 2),
                    "first_wait_p90_s": round(
                        first_waits[min(int(len(first_waits) * .9), len(first_waits) - 1)], 2),
                    "first_wait_max_s": round(first_waits[-1], 2),
                })
    print(json.dumps({"rows": len(trajs), "configs": results}, ensure_ascii=False))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps({"rows": len(trajs), "configs": results}, ensure_ascii=False, indent=2),
            encoding="utf-8")


if __name__ == "__main__":
    main()
