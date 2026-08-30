---
language:
  - zh
  - en
license: apache-2.0
base_model: OpenMOSS-Team/MOSS-TTS-Realtime
pipeline_tag: text-to-speech
---

# MOSS-TTS-Realtime InfiniSST En-Zh v8 phrase

这是为 InfiniSST phrase-policy 输出分布匹配训练的 MOSS-TTS-Realtime checkpoint。

## 训练数据

训练集共 46,753 行，包含 36,529 行 v6 base trajectory、6,385 行完整 phrase trajectory 和 3,839 行 mid-start phrase trajectory。phrase policy 参数为 `multiplier=2`、`chunk_seconds=0.96`、`phrase_max_hold_s=7.68`、`phrase_min_chars=6`。

底座为 `OpenMOSS-Team/MOSS-TTS-Realtime@75682787d8e2fcc73faca37ba2931453ca9c4022`。训练 1 epoch、3 张 H200、global batch 15、learning rate `1e-5`、seed 42，共 3,117 optimizer steps。

`model.safetensors` SHA256：`074964929bce38b9069efc07789336dd231de6d5426554c2662169b610a5e4e9`。

## 评测

在 ACL6060 的 5 个 talk 上，用连续 codec decoder context、Qwen3-ASR、SEGALE 和 SacreBLEU `tokenize=zh` 评测：

| InfiniSST 输入速度 | BLEU | null alignment | 生成音频时长 |
| --- | ---: | ---: | ---: |
| 1× | 34.09 | 1.50% | 3593.84 s |
| 1.5× | 29.61 | 7.14% | 3766.40 s |

匹配 TTS 训练分布后，1.5× 仍下降 4.48 BLEU。因此旧 TTS 分布失配不是跨语速退化的充分解释。这个结论只针对当前 InfiniSST、TTS、ASR 和对齐组成的级联系统，不能单独归因到某一层。

代码、配置和轻量结果位于 [S2S_omni](https://github.com/luojiaxuan/S2S_omni/tree/luojiaxuan/codec-decoder-context-ab) 的 `projects/infinisst_moss_tts_cascade/`。
