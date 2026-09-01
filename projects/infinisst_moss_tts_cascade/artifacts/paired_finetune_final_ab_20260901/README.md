# 配对微调终局 A/B（2026-09-01）

回答用户问题「为什么不直接用原本的 InfiniSST」。两臂均为
"InfiniSST 写出策略 × 与之配对混训的 TTS"，同一 3-talk 1× 子集
（talk 110 / 117 / 268），同一打分管道（逐 turn Qwen3-ASR-1.7B →
SEGALE d0041438 对齐 → SacreBLEU tokenize=zh），BLEU 由同一脚本从
两臂的 `xcomet_input.jsonl` 计算。

| 臂 | InfiniSST | TTS 训练集 | BLEU | 段数 | 备注 |
| --- | --- | --- | --- | --- | --- |
| A′ | 原版（短 delta） | v6 base 36,529 + baseline 轨迹 12,112 混训 | **19.54** | 401 | SEGALE null 85/401（全部 over-translation）；1× 失控 2/5 talk |
| B′ = v8 | phrase（长 delta） | v6 base 36,529 + phrase 轨迹混训 | **33.28** | 318 | SEGALE null 3；失控 0/5 talk |

结论：配对条件对齐后，phrase 线大幅优于原版线（+13.7 BLEU）。
A′ 的死因与 v8 加速档掉分同机制——TTS 失控超额生成——且失控率
跟输入 delta 长度走：短 delta 让失控大增，长 delta 更稳。

## 文件

- `ap_bleu_summary.json` — A′ 的 SEGALE 管道原始 summary
  （hyper00 run `20260830-200824-401864000` `result/ap0/`）。
- `ap_xcomet_input.jsonl` — A′ 逐段对齐产物（BLEU 输入）。
- `aprime_subset_bleu.json` / `bprime_v8_subset_bleu.json` —
  同一脚本对两臂的 BLEU 复算（B′ 从 v8 c2 全量 run 过滤同 3 talk）。

## 产物位置

- A′ TTS 权重：tilde `~/sglang-omni-tts/outputs/model_v10base/checkpoint-epoch-0`
  （仅本地，可由 `data/train_v10base.jsonl` + sft.py + seed 复现；未上 HF）。
- A′ 合成音频与 ASR：hyper00 `/data02/jaxan/ap_score/`、
  run root `eval/output/ap0/`。
- v8 TTS 权重（B′）：HF `gavinlaw/moss-tts-realtime-infinisst-en-zh-v8-phrase@521e09fa`。
