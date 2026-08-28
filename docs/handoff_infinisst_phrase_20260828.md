# InfiniSST phrase-boundary：6 小时窗口交接（2026-08-28）

## 0. 一句话

phrase-boundary 的**策略侧改造已完成并在跑评测**（推理档 `--phrase-merge`，
与重训模型的 write 行为等价）；**InfiniSST 重训的代码改造也已完成**
（collator 补丁 + 数据核验通过），但**训练环境有一串版本坑，最后卡在
flash_attn ABI**，未能在窗口内开训。下面是两条线各自的确切状态。

## 1. 数据核验（用户要求的抽样检查）——全部通过

历史上"multiplier 合并出错导致末尾 chunk 音文错位"的根因已定位：
`model/model.py:224` 的 `validate()`（断言 `len(trajectory)` == 音频 chunk 数）
**被注释掉了**，一旦数据违反该不变量就会末尾错位。核验结果：

| 检查 | 结果 |
| --- | --- |
| 2 万行 `len(trajectory) == n_frames/15360` | **0 违例** |
| 各 multiplier（1/2/4/9/12）文本 chunk 数 vs 音频 chunk 数 | **0 不一致** |
| phrase 重分配前后文本逐字守恒 | **0 例不守恒** |
| phrase 重分配前后 chunk 数 | **0 例变化** |

抽样看末尾 3 个 chunk 的音频区间↔文本，hold 到句末标点才吐，行为正确。

**设计上杜绝了该 bug**：改动只在**同一张 chunk 网格上重新分配文本**，
chunk 数量与每 chunk 的音频 patch 数一字不动。

## 2. InfiniSST 重训：代码已改完，环境卡住

### 已完成的代码改造（taurus `~/InfiniSST`，纯 SST 线，**无 RAG / 无 term map**）

- `train/main.py`：新增 `--phrase_boundary / --phrase_max_hold_s / --phrase_min_chars`
- `model/model.py`：透传给 collator（2 处调用点）
- `train/dataset.py`：新增 `_phrase_redistribute()`；在
  `DataCollatorForTrajectoryInstructMultiLatencyQwenDataset.__call__` 的
  multiplier 合并之后调用。未到短语边界的 chunk 目标置空串（模型学会 hold），
  到边界或超 `max_hold` 才写出。`max_hold` 以**秒**换算，跨 multiplier 行为一致。
- 启动脚本：`~/InfiniSST/train_phrase.sh`（8× A6000，从现役 stage1+stage2
  权重继续训，80k 行子集 `train_phrase80k.tsv`）

### 环境坑（依次踩到，前 3 个已修）

1. ~~`PYTHONPATH` unbound~~ → 已修
2. ~~fairseq 解析到 py3.10 不兼容的 FBK-fairseq~~ → 改用
   `/mnt/aries/data6/jiaxuanluo/fairseq-0.12.2`（有 InfiniSST 需要的
   `read_from_stored_zip`），并修了 FBK-fairseq 的 `collections` 导入
3. ~~fairseq 根目录 `train.py` 抢占 `train` 包名~~ → 补了
   `~/InfiniSST/train/__init__.py`，并把仓库路径置于 PYTHONPATH 首位
4. **未解决**：env 的 `transformers` 是 4.57.1，而 InfiniSST 的
   `patch_llama31/patch_qwen25` 需要 4.46 时代的
   `LlamaFlashAttention2 / Qwen2FlashAttention2`。旁路安装了
   `/mnt/gemini/data2/jiaxuanluo/tf446`（transformers 4.46.3），
   **但随即撞到 `flash_attn_2_cuda` 与 torch 2.7 的 ABI 不匹配**
   （`undefined symbol: _ZN3c105Error...`）。

### 下一步（三选一，按代价排序）

1. 装一个匹配 torch 2.7/cu126 的 flash-attn wheel 到 `tf446` 目录，
   PYTHONPATH 前置后重跑 `train_phrase.sh`（预计 10–20 分钟）；
2. 或改用 `--use_flash_attn False` 并把 patch 里的 FlashAttention2 导入
   改为条件导入（需要小改 `model/patches/patch_qwen25.py`）；
3. 或在 babel 上跑（原配方的环境在那里，路径都是 `/compute/babel-*`）。

启动命令（环境修好后直接可用）：

```bash
ssh taurus 'cd ~/InfiniSST && PYTHONPATH=$PWD:/mnt/gemini/data2/jiaxuanluo/tf446:/mnt/aries/data6/jiaxuanluo/fairseq-0.12.2 nohup bash train_phrase.sh > phrase_train.log 2>&1 &'
```

## 3. 策略侧改造（已完成，评测进行中）

在 InfiniSST 重训之前，先用**等价的输出侧策略**验证收益：
`scripts/moss_multiturn_infer.py` 新增 `--phrase-merge`（默认关闭）——
把 InfiniSST 的增量攒到短语边界（含逗号顿号等句内标点）、≥6 字才成一个
TTS turn，最多 hold 8 个增量。这与重训后模型的 write 行为等价（同一批
增量、同样的 hold 规则），延迟口径也一致，区别只在译文内容不会随策略
重新措辞。评测队列模式：`slidingsoft3phr`。

### 策略模拟（真实训练 manifest 3000 行）

| 配置 | 增量数 | 中位字 | ≤5 字 | 结尾在标点 | 额外延迟 中位/p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 现役 mult=2 | 33,616 | 7 | 31.3% | 36.1% | — |
| phrase（mult=1，hold≤8） | 16,124 | 15 | 1.3% | 95.6% | 1.92s / 5.76s |

## 4. 资源与位置

- InfiniSST 训练数据：`/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv`
  （243,065 行）+ 子集 `train_phrase80k.tsv`
- 现役权重：`/mnt/gemini/data2/jiaxuanluo/stage{1,2}_M=12_norm0_qwen2.5-7b-instruct_rope.bin`
- 评测：hyper00 容器 `sglang-omni-jaxan-phr-1`（GPU0 H200），
  产物 `/data04/jaxan/S2S_omni_runs/moss_tts_infinisst_v2_20260804/`
- taurus 8× A6000 当时全空；hyper01 被他人占满
