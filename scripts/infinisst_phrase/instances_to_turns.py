#!/usr/bin/env python
# note (luojiaxuan): SimulEval instances.log -> 每 talk 一个 swrow turn 流。
# turn 边界 = delays 中相同取值的连续段（一个 write 对应一个 chunk 时刻）。
# note (luojiaxuan): 三条硬断言（字符数守恒 / delay 单调 / 拼接文本逐字相等）
# 是防线，不是装饰——2026-08-28 的 llama31 乱码就是被第一条挡下来的。
# 任何一条不成立直接抛错，不做跳过或近似。
import json, os, sys

inst, outdir, tag = sys.argv[1], sys.argv[2], sys.argv[3]
TALKS = [110, 117, 268, 367, 590]
os.makedirs(outdir, exist_ok=True)

n_turn_total = 0
for line in open(inst, encoding='utf-8'):
    d = json.loads(line)
    idx, pred, delays = d['index'], d['prediction'], d['delays']
    assert len(pred) == len(delays), \
        f"instance {idx}: chars {len(pred)} != delays {len(delays)}"
    assert all(b >= a for a, b in zip(delays, delays[1:])), \
        f"instance {idx}: delays 非单调"
    turns, cur, cur_d = [], [], None
    for ch, dl in zip(pred, delays):
        if cur_d is not None and dl != cur_d:
            turns.append((''.join(cur), cur_d)); cur = []
        cur.append(ch); cur_d = dl
    if cur:
        turns.append((''.join(cur), cur_d))
    assert ''.join(t for t, _ in turns) == pred, f"instance {idx}: 文本不守恒"

    talk = TALKS[idx]
    segs = [{'id': f't{talk}{tag}s{i:04d}', 'text': t, 'delay_ms': dl}
            for i, (t, dl) in enumerate(turns)]
    row = {'row_id': f'talk{talk}_{tag}_full', 'split': 'acl', 'segments': segs}
    with open(f'{outdir}/talk{talk}.{tag}.swrow.jsonl', 'w', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    n_turn_total += len(turns)
print(f'OK {inst} -> {outdir}  talks={len(TALKS)} turns={n_turn_total}')
