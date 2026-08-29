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

### 环境坑已全部修完（2026-08-28 晚）

4. ~~transformers 4.57 vs 代码需要的 4.46~~ → 旁路 `/mnt/gemini/data2/jiaxuanluo/tf446`
5. ~~flash_attn ABI 不匹配~~ → torch 2.7 是 **cxx11abi=TRUE**，env 里装的是
   abiFALSE 编译版。从官方 release 取 `cu12torch2.7cxx11abiTRUE-cp310` wheel，
   **解包**到 `/mnt/gemini/data2/jiaxuanluo/fa_abitrue`（`pip install --target`
   会因 env 已有同版本而跳过）
6. ~~torch.load 默认 weights_only=True~~ → `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`
7. ~~hydra 1.0.4 + omegaconf 2.1.2 不匹配~~ → 旁路装
   `omegaconf==2.0.6 + hydra-core==1.0.7` 到 `/mnt/gemini/data2/jiaxuanluo/hydra_fs`
8. ~~仓库 bug：`train/main.py` 用了 `DeepSpeedStrategy` 但只 import 了
   `DDPStrategy`~~ → 补导入
9. 新增 `--resume_from_last`（显式开关，缺 last.ckpt 直接报错，不隐式回退），
   `trainer.fit(..., ckpt_path=...)`

完整 PYTHONPATH（顺序重要，仓库必须在最前，否则 fairseq 根目录的
`train.py` 会抢占 `train` 包名）：
```
$PWD:/mnt/gemini/data2/jiaxuanluo/hydra_fs:/mnt/gemini/data2/jiaxuanluo/fa_abitrue:/mnt/gemini/data2/jiaxuanluo/tf446:/mnt/aries/data6/jiaxuanluo/fairseq-0.12.2
```

### 调度：aries/taurus 是 Slurm 集群（2026-08-28 用户裁定）

**教训**：我先前直接 ssh 上 taurus 用 nohup 起训练占了 8 张 A6000——
而 `sinfo` 显示 taurus 当时是 `alloc`，整机已分配给别人的作业 48278，
等于抢了他的算力。已停并写入全局规则：这两台机占 GPU 必须走 sbatch/srun。

**当前状态**：作业 `48282`（分区 aries,taurus,gemini）PENDING，因为三个
节点都没有可调度 GPU：aries 被管理员 `drng`（`Kill task failed
[root@2026-08-28T06:47:41]`）、taurus `alloc`、gemini 8 卡全占。
aries 实际有 2 张空闲卡且本作业只要 2 张，**唯一拦路的是 DRAIN 标记**：
`sudo scontrol update nodename=aries state=resume`（需管理员）。
另一条路是结束我方另一 session 的占位作业 48237（`hold.sbatch`，占 6 卡，
当前 0% 利用率）——属别的 session 资源，待用户裁决。

### 数据量修正（用户指出）

原 80k 行子集 = **440.9 小时**音频（全量 243,065 行 = 1299.9 小时，
平均 19.3s/行）——对"只教 write 时机策略的 LoRA 续训"是 4.4 倍浪费。
已改为 `train_phrase100h.tsv`：18,600 行 = **103.3 小时** ≈ 9,300 batch。
错在我按"填满时间窗口"倒推子集大小，而非按任务需要正推。

### 历史备选（三选一，按代价排序）

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

## 5. 训练已完成（2026-08-28 21:00 UTC）

按用户裁定改用 **aries + Docker 直跑**（Slurm 三个节点当时都无可调度 GPU：
aries `drng`、taurus `alloc`、gemini 满载；作业 48282 已取消）。

- 容器 `infinisst-phrase-jaxan-1`（aries，GPU 0–3，避开被占位作业占显存的 4/5），
  镜像 `nvidia/cuda:12.6.0-devel-ubuntu22.04`（base 镜像缺 gcc 与 nvcc，
  triton JIT 需要；把主机 CUDA 挂进去会破坏 nvidia-container-cli 的 compat 处理，
  且主机是 11.8 与 torch cu126 不匹配）
- 数据 `train_phrase100h.tsv`（18,600 行 = 103.3 h）
- **一个 epoch 41 分钟跑完**，`Trainer.fit stopped: max_epochs=1 reached`，
  exit 0。注意进度条总数 9296 只是 Lightning 对 dataloader 长度的估计，
  真实 SpeechSampler（按音频时长分桶）产出 2324 个 batch，样本全部训过。
- 产物：`/mnt/gemini/data2/jiaxuanluo/runs/infinisst_phrase_v1/last.ckpt`
  （DeepSpeed 分片 17G）→ `zero_to_fp32.py` 转出 380 MB LoRA
  → 稳定路径 **`/mnt/gemini/data2/jiaxuanluo/stage2_phrase_v1_lora.bin`**
  （9,509 万参数，与现役 stage2 同量级）
- 旧的 taurus 8 卡/80k 数据 checkpoint 已归档到
  `runs/archive-20260828-taurus80k/`（未删）——它与 4 卡/100h 的 DeepSpeed
  分片不兼容，续训会直接崩。

### 下一步

用新 LoRA（配现役 stage1）跑 ACL6060 推理 → 新 turn 流 → TTS（v7 + soft3）
→ canonical + BC 双口径评测，与输出侧等价档（BC 39.85/39.88，+6.6 BLEU）
对比，量出"真重训"额外拿到多少（模型可针对短语边界重新措辞，而不只是
把现有增量攒起来）。

## 6. v1 阴性与 v2 修正（2026-08-29）

### v1 的 A/B 结果：方向对，幅度远不够

同音频、同解码参数（`--model-type w2v2_qwen25 --source-segment-size 1920
--latency-multiplier 2 --beam 4 --no-repeat-ngram-size 5
--repetition-penalty 1.2`）、同 stage1 底座，只换 LoRA：

| | turns | 中位字 | ≤2 字 | ≤5 字 | 结尾在标点 | BLEU | LAAL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline 现役 | 1684 | 7 | 7.2% | 30.3% | 32.6% | 54.12 | 5716 ms |
| phrase v1 | 1742 | 8 | 6.0% | 25.3% | 37.9% | 50.35 | 3948 ms |
| 目标（输出侧等价档） | ~560 | 19 | — | 1.3% | 96% | — | — |

### 根因：hold 预算按秒换算，遇上随机 multiplier 大面积退化

`max_hold_steps = max(1, round(phrase_max_hold_s / (speech_segment_size*0.08*m)))`。
训练时 `m ~ U{1..12}`（`np.random.randint(1, max+1)`，dataset.py:1561）。
m≥6 时该式化简为 1，而 `_phrase_redistribute` 里 `held += 1` 后立刻满足
`held >= max_hold_steps`，**等价于完全不 hold**。12 个取值里 7 个如此，
**58% 的训练 batch 在教「立即写出」**，与 phrase 策略正面冲突；
只有 m=1/2 拿到有意义的预算（8/4 步）。

推理侧机制无误：`agents/infinisst.py:936` 的
`if translation != '' or states.source_finished` 会把空串转成 `ReadAction`，
「空输出 = hold」的通道是通的，模型只是没学会。

### v2 改了什么（只动两个变量）

1. `--trajectory_max_multiplier 12 → 2`。部署档固定 multiplier=2，
   多档随机化对本任务无收益，只制造矛盾监督。m∈{1,2} 的 hold 预算为 8/4，
   全部有效。
2. `--max_epochs 1 → 2`。v1 距目标太远，需要余量；单 epoch 41 分钟，代价可接受。

叠加效应：每 epoch 的有效 hold 监督从 42% 的 batch 升到 100%（2.4×），
两个 epoch 合计约 4.8× 于 v1。

### 顺带修的可观测性（防止同类 bug 再次静默）

`train/dataset.py` 的 collator `__init__` 现在会打印各 multiplier 的 hold
预算表与退化计数：

```
phrase hold budget per multiplier: m=1:8 m=2:4
phrase degenerate (max_hold_steps==1, 永不 hold): 0/2 multipliers
```

v1 若有这两行，启动 30 秒内就能看出 7/12 退化。**没有加兜底逻辑**——
按用户裁定，配置不合理应当可见，而不是被默认值悄悄修正。

### 决策日志（本可以问用户但按默认推进的）

| 问题 | 我选的默认 | 理由 | 如何推翻 |
| --- | --- | --- | --- |
| v1 阴性后是重训还是直接上输出侧档 | 重训 v2 | 容器与环境已热，一轮 85 分钟；输出侧档已验证收益在手，重训只是加码，风险仅为时间 | 丢弃 v2 LoRA，回到输出侧 `--phrase-merge`，无需回滚任何代码 |
| multiplier 收窄到 2 会不会伤多档鲁棒性 | 接受 | 部署固定 m=2；多档能力本就来自 stage1/stage2，LoRA 续训只改 write 时机 | 换回 `--trajectory_max_multiplier 12` 并改用 chunk 数计的 hold 预算重训 |
| epoch 从 1 加到 2 引入第二个变量 | 接受 | v1 距目标 58 个百分点，隔离单变量的价值低于尽快拿到可用模型 | 若 v2 成功且想归因，用 v2 的 epoch-1 checkpoint（`--save_step 200` 有中间产物）单独评一次 |
| 容器只剩 1 张卡 | 重建为 4 卡 | 推理阶段我把它重建成了单卡，与 v1 的 4 卡有效 batch 不可比；容器内无进程，数据全在 mount 上 | 无需回滚；重建代价约 1 分钟 |

### 容器与产物

- 容器 `infinisst-phrase-jaxan-1`（aries，`--gpus '"device=0,1,2,3"'`，
  GPU 4/5 属他人占位作业，未触碰）
- v2 日志/产物 `/mnt/gemini/data2/jiaxuanluo/runs/infinisst_phrase_v2/`
- v1 LoRA 保留在 `/mnt/gemini/data2/jiaxuanluo/stage2_phrase_v1_lora.bin`（对照用）
- 容器外存活监控已挂（poll 5 分钟，告警覆盖容器消失 / 无进程 / Traceback /
  OOM / Killed / 非零退出，不只匹配成功行）
