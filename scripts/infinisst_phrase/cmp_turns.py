#!/usr/bin/env python
# note (luojiaxuan): 比较各档 turn 结构。结尾标点率是 phrase 策略是否装上的主判据。
import glob, json, statistics, sys, unicodedata

PUNCT = "。！？!?…，,、；;：:"
for spec in sys.argv[1:]:
    name, d = spec.split('=', 1)
    lens, tail, n = [], 0, 0
    for f in sorted(glob.glob(f'{d}/*.jsonl')):
        for s in json.loads(open(f, encoding='utf-8').readline())['segments']:
            t = s['text'].strip()
            if not t:
                continue
            n += 1
            lens.append(sum(1 for c in t if unicodedata.category(c)[0] in ('L', 'N')))
            tail += t[-1] in PUNCT
    le2 = sum(1 for x in lens if x <= 2) / len(lens) * 100
    le5 = sum(1 for x in lens if x <= 5) / len(lens) * 100
    print(f'{name:18s} turns={n:5d} 中位={statistics.median(lens):5.0f}字 '
          f'≤2字={le2:5.1f}% ≤5字={le5:5.1f}% 结尾标点={tail / n * 100:5.1f}%')
