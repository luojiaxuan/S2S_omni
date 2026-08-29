#!/usr/bin/env python
"""在现役 stage2 与短语 LoRA 之间线性插值，扫「结构收益 vs 鲁棒性损失」。

W(a) = W_base + a * (W_phrase - W_base)。a=0 即现役，a=1 即重训模型。
零训练成本，可连续取点。键集必须与参考逐键相同，不同即抛错。
usage: interp_lora.py <base.bin> <phrase.bin> <alpha> <out.bin>
"""
import sys

import torch

base_p, phr_p, a, out_p = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
base = torch.load(base_p, map_location='cpu', weights_only=True)
phr = torch.load(phr_p, map_location='cpu', weights_only=True)
if set(base) != set(phr):
    raise SystemExit('键集不一致，拒绝插值')
out = {k: (base[k].float() + a * (phr[k].float() - base[k].float())).to(base[k].dtype)
       for k in base}
torch.save(out, out_p)
rel = sum((base[k].float() - out[k].float()).norm().item()
          / max(base[k].float().norm().item(), 1e-9) for k in base) / len(base)
print(f'alpha={a}  keys={len(out)}  相对现役平均变化 {rel * 100:.3f}%  -> {out_p}')
