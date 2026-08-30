#!/usr/bin/env python3
"""对比两种短语释放规则的延迟与结构，纯 CPU，不需要模型。

A（现行）：只检查整个 buffer 的**尾字符**是否为标点。buffer 里已有标点但
后面又跟了新字时不释放，例如 `政策中还有更多内容。我` 会继续等。
B（候选）：释放到 buffer 内**最后一个满足 min_chars 的标点**，标点之后的
suffix 留在 buffer 继续攒。

延迟口径：chunk i 写出时，第 j 个 chunk 进入 buffer 的字符等待
(i-j)*step 秒。「首块等待」指该次 write 中最早那个 chunk 的等待。
usage: ablate_phrase_release_rule.py --trajectory-tsv T --ids-jsonl C --multiplier 1
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
SENT_FINAL = "。！？!?…"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory-tsv", required=True, type=Path)
    p.add_argument("--ids-jsonl", required=True, type=Path,
                   help="只取该文件里出现的 id（即实际用于训练的行）")
    p.add_argument("--id-prefix", default="traj_",
                   help="ids-jsonl 的 id 相对 TSV 多出的前缀")
    p.add_argument("--multiplier", type=int, default=1)
    p.add_argument("--chunk-seconds", type=float, default=0.96)
    p.add_argument("--phrase-max-hold-s", type=float, default=7.68)
    p.add_argument("--phrase-min-chars", type=int, default=6)
    p.add_argument("--examples", type=int, default=10)
    p.add_argument("--out-json", type=Path)
    return p.parse_args()


def alnum(s: str) -> int:
    return sum(c.isalnum() for c in s)


def release_index_b(buffer: str, min_chars: int) -> int:
    """B 规则：返回可释放前缀的长度（含标点），无可释放点返回 0。"""
    cut = 0
    for i, c in enumerate(buffer):
        if c in PHRASE_PUNCT and alnum(buffer[: i + 1]) >= min_chars:
            cut = i + 1
    return cut


def simulate(grouped: list[str], rule: str, max_hold: int, min_chars: int):
    """返回 [(write_text, first_chunk_idx, write_idx, [(char_count, chunk_idx)...])]"""
    writes = []
    buf = ""
    parts: list[tuple[int, int]] = []   # (字符数, 该段文本来自哪个 chunk)
    held = 0
    for idx, text in enumerate(grouped):
        buf += text
        if text:
            parts.append((len(text), idx))
        stripped = buf.strip()
        if not stripped:
            held = 0
            continue
        held += 1
        last = len(grouped) - 1
        if rule == "A":
            fire = (stripped[-1] in PHRASE_PUNCT and alnum(buf) >= min_chars) \
                or held >= max_hold or idx == last
            cut = len(buf) if fire else 0
        else:
            cut = release_index_b(buf, min_chars)
            if not cut and (held >= max_hold or idx == last):
                cut = len(buf)
        if not cut:
            continue
        emitted, rest = buf[:cut], buf[cut:]
        # 把 parts 按 cut 切开，前半记入本次 write，后半留给下一次
        used, remain, taken = [], [], cut
        for n, cidx in parts:
            if taken <= 0:
                remain.append((n, cidx))
            elif n <= taken:
                used.append((n, cidx)); taken -= n
            else:
                used.append((taken, cidx)); remain.append((n - taken, cidx)); taken = 0
        writes.append((emitted, used[0][1] if used else idx, idx, used))
        buf, parts = rest, remain
        held = 0 if not rest.strip() else held
        if rest.strip():
            held = 1   # 残留内容已在 buffer 中等待，从本 chunk 起重新计
    if buf.strip():
        writes.append((buf, parts[0][1] if parts else len(grouped) - 1,
                       len(grouped) - 1, parts))
    return writes


def main() -> None:
    a = parse_args()
    step = a.chunk_seconds * a.multiplier
    max_hold = max(1, round(a.phrase_max_hold_s / step))

    keep = set()
    with a.ids_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r'\{"id":\s*"([^"]+)"', line)
            if m:
                i = m.group(1)
                keep.add(i[len(a.id_prefix):] if i.startswith(a.id_prefix) else i)

    csv.field_size_limit(10 ** 8)
    trajs: list[list[str]] = []
    with a.trajectory_tsv.open(newline="", encoding="utf-8") as fh:
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

    report = {"multiplier": a.multiplier, "step_seconds": step,
              "max_hold_steps": max_hold, "min_chars": a.phrase_min_chars,
              "rows": len(trajs), "rules": {}}
    per_rule_examples = {}
    for rule in ("A", "B"):
        char_lat, first_wait, lens, n_tail_sent, n_tail_intra, n_writes = [], [], [], 0, 0, 0
        examples = []
        for t in trajs:
            grouped = ["".join(t[i:i + a.multiplier])
                       for i in range(0, len(t), a.multiplier)]
            for emitted, first_idx, widx, used in simulate(
                    grouped, rule, max_hold, a.phrase_min_chars):
                s = emitted.strip()
                if not s:
                    continue
                n_writes += 1
                lens.append(alnum(s))
                if s[-1] in SENT_FINAL:
                    n_tail_sent += 1
                elif s[-1] in PHRASE_PUNCT:
                    n_tail_intra += 1
                first_wait.append((widx - first_idx) * step)
                for n, cidx in used:
                    char_lat.extend([(widx - cidx) * step] * n)
                if len(examples) < a.examples and rule == "A":
                    examples.append(s)
        char_lat.sort(); first_wait.sort()
        q = lambda v, p: v[min(int(len(v) * p), len(v) - 1)] if v else 0.0
        report["rules"][rule] = {
            "writes": n_writes,
            "char_latency_mean_s": round(statistics.fmean(char_lat), 3),
            "char_latency_p50_s": round(q(char_lat, .50), 3),
            "char_latency_p90_s": round(q(char_lat, .90), 3),
            "first_block_wait_p90_s": round(q(first_wait, .90), 3),
            "first_block_wait_max_s": round(first_wait[-1], 3),
            "median_chars": statistics.median(lens),
            "pct_le5_chars": round(sum(x <= 5 for x in lens) / len(lens) * 100, 2),
            "pct_end_sentence_final": round(n_tail_sent / n_writes * 100, 2),
            "pct_end_intra_sentence": round(n_tail_intra / n_writes * 100, 2),
        }
        per_rule_examples[rule] = examples

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if a.out_json:
        a.out_json.parent.mkdir(parents=True, exist_ok=True)
        a.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")


if __name__ == "__main__":
    main()
