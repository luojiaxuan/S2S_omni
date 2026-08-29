#!/usr/bin/env python
"""把 Lightning 存出的 LoRA 键名对齐到 agent 的加载格式。

Lightning 把 SpeechLLM 挂在 self.model 上，checkpoint 因此多一层 `model.`
前缀；而 agents/infinisst.py 用 `self.model.load_state_dict(..., strict=False)`
加载，前缀不匹配的键会被**静默丢弃**（返回的 missing/unexpected 被忽略），
结果是跑了一个没有适配器的模型却毫无报错。

本脚本剥掉该前缀，并强制要求结果键集与参考文件（现役 stage2）逐键相同，
不相同直接抛错——这是防止同类静默失败复发的唯一保障。
usage: strip_lightning_prefix.py <in.bin> <out.bin> <reference.bin>
"""
import sys

import torch

src, dst, ref = sys.argv[1], sys.argv[2], sys.argv[3]
sd = torch.load(src, map_location='cpu', weights_only=True)
out = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in sd.items()}
ref_keys = set(torch.load(ref, map_location='cpu', weights_only=True).keys())
if set(out.keys()) != ref_keys:
    only_out = sorted(set(out) - ref_keys)[:3]
    only_ref = sorted(ref_keys - set(out))[:3]
    raise SystemExit(f'键集不匹配: 仅在输出 {only_out} ... 仅在参考 {only_ref}')
torch.save(out, dst)
print(f'OK {len(out)} keys -> {dst}')
