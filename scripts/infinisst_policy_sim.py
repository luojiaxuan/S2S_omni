#!/usr/bin/env python3
"""在真实 InfiniSST 训练数据上模拟 phrase-boundary write 策略。

回答训练前必须先知道的问题：把 write 限制在短语边界、并合并过短增量之后，
(a) 增量长度分布变成什么样，(b) 代价是多少延迟（以 0.96s 格为单位）。

策略（都在 0.96s 格的粒度上决定"这一格 write 还是 hold"）：
  base       现状：每 mult 格无条件 write（内容为这 mult 格的拼接）
  punct      只在累积文本以句内/句末标点收尾时 write，否则 hold；
             hold 超过 max_hold 格则强制 write（延迟上限）
  punct_min  punct 之外再要求 ≥min_chars 字，否则继续 hold
延迟口径：一个字的延迟 = 它本可被 write 的格 → 实际被 write 的格，
差值 × 0.96s。base 也有该延迟（等 mult 对齐），所以比较的是增量。
"""
import argparse
import ast
import csv
import statistics
import sys
import unicodedata

PUNCT_ALL = '。！？!?…，,、；;：:'
PUNCT_SENT = '。！？!?…'

ap = argparse.ArgumentParser()
ap.add_argument('--tsv', required=True)
ap.add_argument('--mult', type=int, default=2, help='部署 latency multiplier（1 格=0.96s）')
ap.add_argument('--max-rows', type=int, default=3000)
ap.add_argument('--max-hold', type=int, default=4, help='最多额外 hold 几个 mult 步')
ap.add_argument('--min-chars', type=int, default=6)
args = ap.parse_args()

csv.field_size_limit(10 ** 8)


def spoken(t):
    return sum(1 for ch in t if unicodedata.category(ch)[0] in ('L', 'N'))


def load(path, limit):
    out = []
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
            raw = (r.get('trajectory') or '').lstrip()
            if not raw.startswith('['):
                continue
            try:
                traj = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                continue
            traj = [t[0] if isinstance(t, (list, tuple)) else t for t in traj]
            if traj:
                out.append(traj)
            if len(out) >= limit:
                break
    return out


def simulate(traj, mult, policy, max_hold, min_chars):
    """返回 [(增量文本, 该增量被 write 的步 index, 内容最早可 write 的步 index)]"""
    steps = [''.join(traj[i:i + mult]) for i in range(0, len(traj), mult)]
    writes, buf, buf_origin = [], '', None
    for s_idx, s in enumerate(steps):
        if s and buf_origin is None:
            buf_origin = s_idx
        buf += s
        if not buf.strip():
            continue
        if policy == 'base':
            ok = True
        else:
            tail = buf.rstrip()
            at_punct = bool(tail) and tail[-1] in PUNCT_ALL
            long_enough = spoken(buf) >= min_chars if policy == 'punct_min' else True
            forced = (s_idx - buf_origin) >= max_hold
            ok = (at_punct and long_enough) or forced
        if ok:
            writes.append((buf, s_idx, buf_origin))
            buf, buf_origin = '', None
    if buf.strip():
        writes.append((buf, len(steps) - 1, buf_origin if buf_origin is not None else len(steps) - 1))
    return writes


rows = load(args.tsv, args.max_rows)
print(f'样本行数 {len(rows)}  mult={args.mult}（每步 {args.mult * 0.96:.2f}s）'
      f'  max_hold={args.max_hold} 步  min_chars={args.min_chars}\n')
print(f'{"策略":11s} {"增量数":>7s} {"中位字":>6s} {"≤2字":>6s} {"≤5字":>6s} '
      f'{"末尾有标点":>9s} {"末尾句末标点":>11s} {"额外延迟(中位/p90)":>18s}')
for policy in ('base', 'punct', 'punct_min'):
    lens, ends_p, ends_sent, delays, n = [], 0, 0, [], 0
    for traj in rows:
        for text, w_idx, o_idx in simulate(traj, args.mult, policy, args.max_hold, args.min_chars):
            n += 1
            lens.append(spoken(text))
            tail = text.rstrip()
            if tail and tail[-1] in PUNCT_ALL:
                ends_p += 1
            if tail and tail[-1] in PUNCT_SENT:
                ends_sent += 1
            delays.append((w_idx - o_idx) * args.mult * 0.96)
    lens_s = sorted(lens)
    d = sorted(delays)
    print(f'{policy:11s} {n:7d} {statistics.median(lens):6.0f} '
          f'{sum(1 for x in lens if x <= 2) / n * 100:5.1f}% '
          f'{sum(1 for x in lens if x <= 5) / n * 100:5.1f}% '
          f'{ends_p / n * 100:8.1f}% {ends_sent / n * 100:10.1f}% '
          f'{statistics.median(d):8.2f}s /{d[int(len(d) * 0.9)]:6.2f}s')
