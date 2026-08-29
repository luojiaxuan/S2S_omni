#!/usr/bin/env python
"""校验 LoRA 键名与参考文件一致，不一致直接非零退出。

agents/infinisst.py 用 strict=False 加载 LoRA 且丢弃返回的 missing/unexpected，
键名对不上时会**静默跑一个没有适配器的模型**——2026-08-29 的 v1/v2 早读就
因此产出了两份逐字节相同的假结果。推理前必须过这道校验。
usage: check_lora_keys.py <lora.bin> <reference.bin>
"""
import sys

import torch

a = set(torch.load(sys.argv[1], map_location='cpu', weights_only=True).keys())
b = set(torch.load(sys.argv[2], map_location='cpu', weights_only=True).keys())
if a != b:
    raise SystemExit(
        f'LoRA 键名与参考不一致：仅在待用文件 {sorted(a - b)[:2]}，'
        f'仅在参考 {sorted(b - a)[:2]}。加载会被静默忽略，拒绝运行。')
print(f'LORA_KEYS_OK {len(a)} keys')
