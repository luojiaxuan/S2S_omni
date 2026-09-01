#!/usr/bin/env python
"""离线合并 LoRA adapter 进 base，产出下游可直接加载的全量 checkpoint。

单进程执行，规避训练内 merge 在多 rank DDP 下的死锁。合并后把推理资产
（configuration/processing/streaming 代码等）拷进目录，布局与全参 checkpoint 一致。
usage: merge_lora.py <base_dir> <ckpt_dir(含 adapter/)>
"""
import shutil
import sys
from pathlib import Path

import torch

base_dir, ckpt_dir = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, '/home/guests/zhen/sglang-omni-tts/code/src')
from moss_tts_realtime.mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
from peft import PeftModel

model = MossTTSRealtime.from_pretrained(str(base_dir), torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, str(ckpt_dir / 'adapter'))
model = model.merge_and_unload()
model.save_pretrained(str(ckpt_dir), safe_serialization=True)
for f in base_dir.iterdir():
    if f.suffix in ('.json', '.txt') or f.name.endswith('.py') or f.is_dir():
        dst = ckpt_dir / f.name
        if not dst.exists():
            (shutil.copytree if f.is_dir() else shutil.copy2)(f, dst)
print(f'MERGED {ckpt_dir}')
