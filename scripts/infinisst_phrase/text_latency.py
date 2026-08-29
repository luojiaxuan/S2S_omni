#!/usr/bin/env python
"""比较各策略下"译文文本何时可用"的延迟，不依赖 ASR 与外部 API。

每个 turn 自带 delay_ms（该 turn 文本产生的时刻）。把 turn 内每个字符都记为
在该时刻可用，按字符加权求平均，即得"平均文本可用延迟"。同一段音频、同一
参考，所以三档可直接比较；同时报总字数，防止"少输出内容换低延迟"。

输出侧规则档没有独立的 turn 流（合并发生在 TTS 脚本内），故在 baseline 增量
上精确复现其合并逻辑：攒到句内标点且 >= min_chars 字，或攒满 max_hold 个增量
才吐出，吐出时刻取该组最后一个增量的时刻。
usage: text_latency.py <名称=turns目录> ...
"""
import glob
import json
import statistics
import sys
import unicodedata

PUNCT = "。！？!?…，,、；;：:"


def alnum(s):
    return sum(1 for c in s if unicodedata.category(c)[0] in ('L', 'N'))


def phrase_merge(segs, min_chars=6, max_hold=8):
    out, buf, held = [], [], 0
    for s in segs:
        t = s['text']
        if not t.strip():
            continue
        buf.append(s)
        held += 1
        joined = ''.join(x['text'] for x in buf).strip()
        at_boundary = joined[-1] in PUNCT and alnum(joined) >= min_chars
        if at_boundary or held >= max_hold:
            out.append({'text': joined, 'delay_ms': buf[-1]['delay_ms']})
            buf, held = [], 0
    if buf:
        out.append({'text': ''.join(x['text'] for x in buf),
                    'delay_ms': buf[-1]['delay_ms']})
    return out


def stats(segs):
    w = [(alnum(s['text']), s['delay_ms']) for s in segs if alnum(s['text'])]
    total = sum(n for n, _ in w)
    mean = sum(n * d for n, d in w) / total
    per_char = sorted(d for n, d in w for _ in range(n))
    return total, mean, per_char[int(len(per_char) * 0.9)], len(w)


print(f"{'配置':26s} {'turn 数':>7s} {'总字数':>7s} {'平均可用延迟':>12s} {'p90':>9s}")
for spec in sys.argv[1:]:
    name, d = spec.split('=', 1)
    merge = name.endswith('[merge]')
    segs = []
    for f in sorted(glob.glob(f'{d}/*.jsonl')):
        segs += json.loads(open(f, encoding='utf-8').readline())['segments']
    if merge:
        segs = phrase_merge(segs)
        name = name[:-7]
    total, mean, p90, n = stats(segs)
    print(f'{name:26s} {n:7d} {total:7d} {mean / 1000:9.2f} s {p90 / 1000:6.2f} s')
