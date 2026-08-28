# InfiniSST phrase-boundary SFT → TTS 重训：准备工作（2026-08-28）

## 0. 一句话

音色跳变的根因是 InfiniSST 吐出的增量太碎、且切在短语中间；修法是重训
InfiniSST 让 write 只落在短语边界并合并过短增量，再用新 InfiniSST 的输出
文本重训 TTS。本文是动手前的准备：数据在哪、怎么改、怎么训、TTS 怎么接。

## 1. 因果链与已测证据

- TTS 侧每个增量都要独立起音/收尾，声学状态每轮重置 → 接缝处音色、基频
  漂移。11 分钟 talk 有 ~350 个 turn，等于每 2 秒一条接缝。
- 推理侧句级缓冲（`--sentence-merge`，台账 4.-21f）证实了这条因果链：
  turn 数 350 → 71–95，边界受控 BLEU 34.11 → 41.68、漏译归零。但那是事后
  补救，代价是攒句延迟。
- **碎片化实测**（lab trajectory TSV，800 行，jieba 判词边界）：

  | 部署档 | 增量中位字数 | ≤2 字 | ≤5 字 | 切在词中间 | 结尾无标点 |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | mult=1（0.96s） | 4 | 28.7% | 67.2% | 0.0% | 73.7% |
  | **mult=2（1.92s，现役）** | **7** | **11.9%** | **35.1%** | **0.0%** | **69.7%** |
  | mult=9（8.64s） | 29 | 0.9% | 2.5% | 0.0% | 55.0% |

  **诊断精确命中**：trajectory 是按词对齐切的，所以从不切断词；但它只看
  源侧时间，不看目标侧句法——所以 70% 的增量结尾不在任何标点处，即切在
  短语中间。

## 2. 数据在哪（taurus，已核实存在）

| 用途 | 路径 | 规模 |
| --- | --- | --- |
| **训练 manifest** | `/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv` | 374 MB，**243,065 条** |
| dev manifest | `/mnt/gemini/data1/jiaxuanluo/dev_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv` | 356 条 |
| 原始音频 | `/mnt/taurus/data/siqiouyang/datasets/gigaspeech/audio/` | — |
| MFA 词级对齐 | `/mnt/taurus/data/siqiouyang/datasets/gigaspeech/textgrids/` | 8,244,757 个 TextGrid |
| MFA 索引 | `/mnt/gemini/data1/jiaxuanluo/gigaspeech_mfa_index/gigaspeech_mfa_index.sqlite` | 2.34 GB |
| **现役 en→zh 权重** | `/mnt/gemini/data2/jiaxuanluo/stage1_M=12_norm0_qwen2.5-7b-instruct_rope.bin`（1.2 GB，encoder+adapter）<br>`…/stage2_M=12_…_rope.bin`（363 MB，LoRA r32） | GigaSpeech stage2 产物 |
| 服务副本 | `/mnt/aries/data6/jiaxuanluo/demo/en-zh/{pytorch_model.bin,lora.bin}` | 字节一致 |

manifest 字段（11 列，tab，`QUOTE_NONE`）：
`id | audio | n_frames | speaker | src_lang | tgt_lang | src_trajectory | asr | src_text | tgt_text | trajectory`
——`trajectory` 每项对应 **0.96s** 音频（`n_frames / 15360 = len(trajectory)`），
空串=等待；`src_trajectory` 是同粒度的英文增量（做边界判据的额外原料）。

**注意**：仓库脚本里的 `ROOT` 全部指向 CMU babel 路径（不可达）；真实可用的
是上表路径，训练脚本需改 `--data_path` 与 split 名。

## 3. 怎么改（最小面：只改合并规则，不重建数据）

**结论：不需要重跑 MFA、不需要重建 243k 行 trajectory。** 0.96s 格保持
原样作原子单元，只改"哪些格边界成为 write"。

- 落点：`~/InfiniSST/train/dataset.py:1383-1391`
  （`DataCollatorForTrajectoryInstructMultiLatencyQwenDataset.__call__`），
  现状是把连续 `multiplier` 个格无条件 `''.join` 成一个 write。
- 改成：**累积到短语边界才 write，否则该步 write 空串（=hold）**；
  hold 超过 `max_hold` 步强制 write（延迟上限）；过短的增量继续并入下一步。
- 判据：累积文本以标点收尾（`。！？…，、；：`）且发音字数 ≥ `min_chars`。
  原料现成，不需要额外模型。
- **loss mask 已有**：`dataset.py:1483` 每个 chunk 第二元素是 bool
  "是否计 loss"（`prob_aug` 的 shift/merge 机制在用），可直接复用。
- **参考实现**：`dataset.py:1393-1421` 的 `prob_aug` 分支已有 jieba 的
  shift/merge 逻辑（zh 专用，当前 `--trajectory_prob_aug 0.0` 未启用），
  是现成模板。
- **推理侧零改动**：`agents/infinisst.py:936` 已是
  `if translation != '' or states.source_finished:`——模型输出空串即 hold，
  现有 SimulEval agent 直接兼容。策略因此是**学出来的**，不是硬编码的。

### 策略模拟（真实训练 manifest 前 3000 条）

| 配置 | 增量数 | 中位字 | ≤2 字 | ≤5 字 | 结尾有标点 | 额外延迟 中位/p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mult=2 base（**现役**） | 33,616 | 7 | 9.5% | 31.3% | 36.1% | — |
| mult=2 + punct(hold≤3) | 14,388 | 17 | 1.7% | 7.1% | 84.3% | 1.92s / 5.76s |
| **mult=1 + punct_min(hold≤8)** | **16,124** | **15** | **0.3%** | **1.3%** | **95.6%** | **1.92s / 5.76s** |
| mult=1 + punct_min(hold≤12) | 15,676 | 15 | 0.2% | 1.1% | 98.8% | 1.92s / 5.76s |

**推荐 mult=1 + punct_min，max_hold=8，min_chars=6**：读粒度细化到 0.96s
（比现役 1.92s 更早看到内容，抵掉一部分 hold 延迟），写只在短语边界。
相对现役：增量长度 7 → 15 字（2 倍），≤5 字碎片 31.3% → 1.3%，
结尾在标点 36% → 96%，代价约 +1.5–2s 中位延迟。
比句级缓冲（turn 数降 4 倍）温和，延迟代价小得多。

## 4. SFT InfiniSST

- 入口：`python train/main.py`（Lightning）；配方沿用
  `scripts/train/stage2_gigaspeech_zh_norm0_qwen_rope.sh`：w2v2 冻结、
  LLM LoRA r32、lr 1e-4、`--block_size 48 --max_cache_size 576`
  `--trajectory 9 --trajectory_max_multiplier 12`。
- 初始化：`--sllm_weight_path` = 现役 stage1 bin，`--lora_path` = 现役
  stage2 bin（即从现役权重继续训，不从头）。
- 需改的参数：`--data_path` 指向 `/mnt/gemini/data1/jiaxuanluo/`，
  split 名改为实际文件名；新增 phrase-boundary 相关开关。
- 算力：taurus 8× A6000 48GB，当前 4 张空闲（GPU 4–7）。原配方用 8× L40S；
  LoRA + w2v2 冻结在 4× A6000 上可行，必要时降 `--train_bsz` 提
  `--grad_acc_steps` 保持等效 batch。
- **磁盘是硬约束**：`/mnt/data` 100% 满（剩 101G），`/mnt/gemini/data1`
  剩 361G、`data2` 剩 263G。开训前必须先规划 checkpoint 落盘位置并清理，
  否则会中途写满。

## 5. 再训 TTS（v8）

新 InfiniSST 训完后：
1. 用它在训练语料上重新生成 trajectory（推理跑一遍，得到新的增量切分），
   或直接用改造后的 collator 规则离线重放 trajectory 列——两者等价，
   后者便宜得多，先用后者。
2. 按 v7 既有管线重建 TTS 训练集：`build_moss_rows_from_trajectory.py`
   → 整段合成 → `align_slice_moss_v2.py`（**reject 版**，commit 78526ad）
   → midstart 副本 → 与 v6 沿用行合并。
3. **同时修掉 v7 的两个已知数据缺陷**（台账 4.-21）：合成采样对齐上游
   默认 0.6/30（现为 0.9/50）、slicer 不再插值兜底。
4. 预期收益叠加：训练分布（短语级 turn）与推理分布终于一致——这正是 v7
   在 1.5× 退化的根因（traj 数据是 0.96s 碎切片、推理 turn 更长更密）。

## 6. 待裁决 / 风险

1. **延迟预算**：+1.5–2s 中位是否可接受？若不可接受，可降 `max_hold`
   到 4（结尾在标点降到 ~80%，延迟 p90 降到 3.84s）。**建议先按
   max_hold=8 训，同时保留 max_hold 作为推理期可调项。**
2. **训练时是否随机化 max_hold**：像现有 multiplier 一样随机采样，可得到
   一个"延迟可调"的模型（推理时用不同 hold 换延迟/连贯度）。推荐采用。
3. **磁盘**：开训前必须解决落盘空间。
4. 现役 checkpoint 是 GigaSpeech stage2 产物；lei_lecture 的 stage3/4 权重
   在 taurus 上不存在，若要沿用讲座域适配需从 babel 取。
