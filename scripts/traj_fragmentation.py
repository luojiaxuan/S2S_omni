#!/usr/bin/env python3
"""量化 InfiniSST 输出轨迹的碎片化程度（phrase-boundary 改造的依据）。

输入是 lab 管线 stage-7 的 trajectory TSV（与 InfiniSST 训练 manifest 的
trajectory 同构：每 0.96s 一格，空串=等待）。统计：
  - multiplier=1 与实际部署 multiplier 下的增量长度分布
  - 有多少增量切在词中间（jieba 分词跨界）
  - 有多少增量本身不构成一个可独立发音的短语（≤N 字、无标点收尾）
"""
import ast
import csv
import statistics
import sys
import unicodedata
from collections import Counter

import jieba

TSV = sys.argv[1] if len(sys.argv) > 1 else \
    '/Users/luojiaxuan/Downloads/train_en_utt_robust_asr-filtered_zh_metricx-qe3.0_align.tsv'
MULTIPLIERS = [1, 2, 9]
MAX_ROWS = 800

csv.field_size_limit(10 ** 8)


def spoken(t):
    return sum(1 for ch in t if unicodedata.category(ch)[0] in ('L', 'N'))


rows = []
with open(TSV, newline='') as fh:
    for r in csv.DictReader(fh, delimiter='\t'):
        raw = (r.get('trajectory') or '').lstrip()
        if not raw.startswith('['):
            continue
        try:
            traj = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            continue
        if isinstance(traj, list) and traj:
            rows.append(traj)
        if len(rows) >= MAX_ROWS:
            break

print(f'样本行数 {len(rows)}\n')
for mult in MULTIPLIERS:
    lens, mid_word, no_punct_end, n_inc = [], 0, 0, 0
    for traj in rows:
        # 与 dataset.py 的 collator 同法：连续 mult 格拼一个增量
        merged = [''.join(traj[j] if isinstance(traj[j], str) else traj[j][0]
                          for j in range(i, min(i + mult, len(traj))))
                  for i in range(0, len(traj), mult)]
        merged = [m for m in merged if m.strip()]
        full = ''.join(merged)
        # 整段分词后的词边界字符位置集合
        bounds, pos = set(), 0
        for w in jieba.cut(full):
            pos += len(w)
            bounds.add(pos)
        cur = 0
        for m in merged:
            n_inc += 1
            lens.append(spoken(m))
            cur += len(m)
            if cur not in bounds:
                mid_word += 1
            s = m.rstrip()
            if s and s[-1] not in '。！？!?…，,、；;：:':
                no_punct_end += 1
    lens.sort()
    p = lambda q: lens[int(len(lens) * q)]
    print(f'--- multiplier={mult}（每增量 {mult * 0.96:.2f}s 源音频）')
    print(f'  增量数 {n_inc}，字数 中位 {statistics.median(lens):.0f} / p10 {p(0.1)} / p90 {p(0.9)}')
    print(f'  ≤2 字 {sum(1 for x in lens if x <= 2) / n_inc * 100:5.1f}% |'
          f' ≤5 字 {sum(1 for x in lens if x <= 5) / n_inc * 100:5.1f}%')
    print(f'  切在词中间 {mid_word / n_inc * 100:5.1f}% | 结尾无任何标点 {no_punct_end / n_inc * 100:5.1f}%')
