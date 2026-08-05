# InfiniSST en->zh MOSS-TTS-Realtime Finetuning

## 目标

本任务只做 `en -> zh`。目标不是提升 TTS 拟人程度，而是让
`OpenMOSS-Team/MOSS-TTS-Realtime` 适应 InfiniSST / RASST streaming
speech-to-text 翻译 segment 的中文 token 节奏：短 chunk、半句、跨 chunk
断句都应该被自然读出，而不是按普通完整句 TTS 的节奏强行重断句。

## 已确认接口

MOSS-TTS upstream finetuning 目录：

```text
https://github.com/OpenMOSS/MOSS-TTS/tree/main/moss_tts_realtime/finetuning
```

其 raw JSONL 格式是：

```json
{
  "id": "sample",
  "ref_wav": "source_wavs/train/sample.wav",
  "conversations": [
    {
      "role": "assistant",
      "text": "中文翻译segment",
      "wav": "target_wavs/train/sample.wav"
    }
  ]
}
```

`prepare_data.py` 会把 `ref_wav` 和每个 turn 的 `wav` 编成
`OpenMOSS-Team/MOSS-Audio-Tokenizer` 的 16-codebook `audio_codes`。
`sft.py` 训练时只对 assistant turn 的 audio codes 设 label。

SGLang-Omni PR serving：

```text
https://github.com/sgl-project/sglang-omni/pull/1192
```

当前 PR 提供 `/v1/audio/speech`，支持：

```json
{
  "model": "OpenMOSS-Team/MOSS-TTS-Realtime",
  "voice": "default",
  "input": "中文翻译segment",
  "ref_audio": "/path/to/source.wav",
  "response_format": "wav"
}
```

单个 serving 进程按 one active request 使用；hyper00 当前采用同一 container
内 4 个 serving 进程并行，分别绑定物理 GPU0-3 和端口
`48731,49157,52391,54863`。

## 数据来源

当前使用 plain RASST / InfiniSST baseline zh 数据：

```text
train: /mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline.jsonl
dev:   /mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline_dev.jsonl
```

Taurus 上已检查：

| split | rows | turns | non-empty assistant turns | unique source wav | source wav size |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 12,500 | 70,269 | 62,707 | 62,707 | 6.873 GiB |
| dev | 355 | 891 | 801 | 801 | 0.073 GiB |

这些 JSONL 只有 source speech chunk 和中文 assistant text chunk；没有真实
target speech。因此 target wav 用 MOSS-TTS-Realtime serving 从中文 segment
生成，`ref_audio` 使用对应 source wav。

## 数据包设计

为了避免跨机器传输大量小文件，HF 上传的是 compact source package：

```text
infinisst-moss-tts-en-zh-segments-v1/
  README.md
  dataset_summary.json
  manifest/
    train_segments.jsonl
    dev_segments.jsonl
    train_moss_requests.jsonl
    dev_moss_requests.jsonl
    train_moss_raw_unresolved.jsonl
    dev_moss_raw_unresolved.jsonl
  audio/
    train_source_wavs.tar.zst
    dev_source_wavs.tar.zst
```

其中 `*_moss_requests.jsonl` 是 target wav 生成输入；
`*_moss_raw_unresolved.jsonl` 是 MOSS raw finetuning shape，但
`target_wavs/...` 需要在 hyper00 上生成后才存在。

计划 HF dataset repo：

```text
gavinlaw/infinisst-moss-tts-en-zh-segments
```

当前已上传为 private HF dataset：

```text
repo: https://huggingface.co/datasets/gavinlaw/infinisst-moss-tts-en-zh-segments
commit: c54868f00916e26c7b3893149f2ce43aa13f9632
```

上传内容已在 Taurus 和 hyper00 校验：

| artifact | bytes | sha256 | entries |
| --- | ---: | --- | ---: |
| `audio/train_source_wavs.tar.zst` | 5,988,633,489 | `b14f187af5c7f87788fc6dafa218887641df344457245ff9e041117ba4774fdb` | 62,707 |
| `audio/dev_source_wavs.tar.zst` | 66,275,337 | `ae6f409d6278f5b9173c7a2ccc63f3f0bb6dcc2c7746a3d56de46be7cb15122f` | 801 |

hyper00 local staging:

```text
/data/datasets/infinisst-moss-tts-en-zh-segments-v1
```

该目录已经解压 `source_wavs/train` 和 `source_wavs/dev`：

```text
train source wavs: 62,707
dev source wavs: 801
total local size after extraction: 13G
```

## 生成数据包

在 Taurus 上运行：

```bash
cd /mnt/data/jiaxuanluo/S2S_omni
python scripts/build_infinisst_moss_tts_package.py \
  --train-jsonl /mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline.jsonl \
  --dev-jsonl /mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline_dev.jsonl \
  --output-dir /mnt/data/jiaxuanluo/S2S_omni_data \
  --dataset-id infinisst-moss-tts-en-zh-segments-v1 \
  --compression zstd \
  --zstd-level 6
```

先做 smoke 可以限制数量：

```bash
python scripts/build_infinisst_moss_tts_package.py \
  --output-dir /mnt/data/jiaxuanluo/S2S_omni_data_smoke \
  --max-train-segments 100 \
  --max-dev-segments 20
```

## hyper00 训练流程

先把 HF dataset 下载并解压到 container 内：

```text
/data/datasets/infinisst-moss-tts-en-zh-segments-v1
```

解压 source wav：

```bash
cd /data/datasets/infinisst-moss-tts-en-zh-segments-v1
tar --zstd -xf audio/train_source_wavs.tar.zst
tar --zstd -xf audio/dev_source_wavs.tar.zst
```

启动 MOSS-TTS-Realtime serving 时要允许读取数据目录：

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-TTS-Realtime \
  --config examples/configs/moss_tts_realtime.yaml \
  --allowed-local-media-path /data/datasets/infinisst-moss-tts-en-zh-segments-v1 \
  --port 8000
```

然后运行：

```bash
DATASET_ROOT=/data/datasets/infinisst-moss-tts-en-zh-segments-v1 \
S2S_OMNI_ROOT=/data/S2S_omni \
MOSS_TTS_ROOT=/data/MOSS-TTS \
MOSS_BASE_URL=http://127.0.0.1:8000 \
bash scripts/run_infinisst_moss_tts_hyper00.sh
```

当前 full target generation 使用 4-way launcher：

```bash
DATASET_ROOT=/data/datasets/infinisst-moss-tts-en-zh-segments-v1 \
S2S_OMNI_ROOT=/data/S2S_omni \
RUN_ROOT=/data/S2S_omni_runs/moss_tts_infinisst_20260804_0939 \
PORTS=48731,49157,52391,54863 \
bash scripts/run_moss_target_generation_4way.sh
```

状态检查：

```bash
python /data/S2S_omni/scripts/report_moss_tts_run_status.py \
  --run-root /data/S2S_omni_runs/moss_tts_infinisst_20260804_0939 \
  --dataset-root /data/datasets/infinisst-moss-tts-en-zh-segments-v1
```

该 launcher 会执行：

1. `scripts/generate_moss_realtime_targets.py`
2. `moss_tts_realtime/finetuning/prepare_data.py`
3. `moss_tts_realtime/finetuning/sft.py`

## 2026-08-04 后半段修复记录

上一段交接中 supervisor 卡住的原因已定位并修复，prepare + full SFT 已重新启动。

1. **dev bad_reject_count=2 根因**：`dev_r000031_t000`（"你必须"）和
   `dev_r000031_t001`（"指出来。"）的 ref wav 只有 `0.064s`（1024 samples @16k），
   MOSS-Audio-Tokenizer 只编出一帧 reference codes，serving 报
   `reference codes must be rank 2, got (16,)`。修复方式：用同一 podcast 集
   （`POD1000000010`，clip 242，2.88s）的 `source_wavs/dev/dev_r000030_t000.wav`
   作替代 ref 单独 regenerate 这两条，全部 accepted。原 rejected 行备份在
   `raw/backup_bad_rejects_20260804/`，retry 输入在
   `raw/dev_moss_requests_retry_r000031.jsonl`（metadata 里带
   `ref_wav_original` 和 `ref_wav_substituted_reason`）。修复后：
   train `accepted=62381 rejected=326 bad=0`，dev `accepted=798 rejected=3 bad=0`，
   剩余 reject 全部是纯标点。

2. **TTS runaway 导致 prepare OOM**：第一次重启 prepare_data.py 后 rank2 在
   audio tokenizer attention 处 CUDA OOM（单次分配 128 GiB）。原因是 MOSS
   偶发 runaway，把短 segment 生成到 4096-frame 上限（327.68s @12.5Hz）：
   train 有 89 条 target >30s（其中 23 条顶到 327.7s，文本只有 18-75 字符），
   dev 有 1 条（79.6s，9 字符）。修复：`run_moss_prepare_and_sft_after_generation.sh`
   新增 `filter_split`（commit `e082d4e`），在 coverage validation 之后按
   `MAX_TARGET_DURATION_S`（默认 30s）过滤，audit 记录写
   `raw/{split}_dropped_overlong.jsonl`。本次运行：train kept `62292` / dropped
   `89`，dev kept `797` / dropped `1`。

3. **hyper00 checkout 是 sparse checkout**：只含 `/scripts/`、`/docs/`、
   `/README.md`，导致 `configs/accelerate_ddp_4gpu.yaml` 在磁盘上不存在而
   train 阶段必挂。已 `git sparse-checkout add /configs/` 并 fast-forward 到最新。

4. **supervisor pidfile 陷阱**：`docker exec -d bash -c "cd RUN && ... & echo $! > pids/..."`
   中 `echo` 在 `cd` 生效前的工作目录执行，相对路径写失败，pidfile 残留旧 PID。
   已手工写入真实 PID；下次启动时 pidfile 用绝对路径。

监控（独立于容器）：

- host 侧只读状态脚本：`/data04/jaxan/S2S_omni_runs/monitor_moss_20260804/check_status.sh`
  （输出 alive/stage/stale/prepared/full_ckpts/failure 计数/RAM/GPU/进度一行）。
- Mac 侧 5 分钟轮询该脚本，阶段切换、failure signature、>15min 无日志进展、
  RAM <300G、supervisor 退出时告警。
- `/gpu-utilization-monitor`（10s 窗口，90% 阈值，3 连续低窗告警）同时在跑。

## 2026-08-04 v1 full SFT 完成 + v2 设计讨论

v1 full SFT（per-segment 数据）已完成：

- 第一次 SFT 启动全 rank 崩溃：容器里存在一个残缺的 `flash_attn` 包
  （import 可用但无 `__version__`），`sft.py` 的 `--attn-implementation auto`
  因此解析为 `flash_attention_2`，触发 transformers flash-attn 集成层
  `s_aux=None` 的 `.to()` AttributeError。修复（commit `622c559`）：训练
  默认显式 `ATTN_IMPLEMENTATION=sdpa`（与已验证 smoke 一致），并给
  `prepare_split` 加 skip-existing（rank 输出行数与 filtered 输入吻合即跳过）。
- 训练：4 GPU DDP bf16 sdpa，per-device batch 1，grad accum 4，1 epoch =
  3,893 steps，~0.37s/step（~43 samples/s），约 25 分钟跑完。loss 4.24 ->
  ~3.5-3.7。注意日志后段 `lr=0.00e+00`，LR schedule 可能提前衰减到 0，
  下次 run 检查 scheduler 的 total steps 配置。
- checkpoint：
  `checkpoints/moss_tts_realtime_infinisst_train/checkpoint-epoch-0/`
  （`model.safetensors` 4.66GB）。已上传 HF private model repo：
  `gavinlaw/moss-tts-realtime-infinisst-en-zh`（v1 per-segment baseline，
  provenance 见 repo README）。

**v1 数据的定位（重要）**：v1 target 是 base MOSS 对每个 segment **独立**
生成的（input=中文 segment 文本，ref_audio=对应英文 source chunk），属于
per-segment 自蒸馏。它能教模型接受 segment 形态输入、抑制 fragment runaway，
但监督里没有跨 chunk 韵律连续性——segment 边界处的韵律正是"普通完整句
TTS 节奏"。v1 checkpoint 只作 baseline/消融项。

**v2 设计（luojiaxuan 提出，待执行）**：用完整 row target text 一次调
MOSS-Realtime 合成 long speech reference -> 过 MOSS-Audio-Tokenizer 得连续
codec codes（只编一次）-> 中文 segment 文本对 long speech 做 forced
alignment（文本是 TTS 精确输入，无 ASR 误差；MFA mandarin 或 zh CTC
aligner，只需 segment 边界时间戳）-> 在 codec 帧级（12.5Hz）切片得到每个
segment 的 target codes。这样半句/跨 chunk 边界的监督是"句中"韵律，与
streaming 推理背靠背播放的形态一致。可选升级：一个 row 写成一条多
assistant turn 记录，生成第 k 段时能看到前 k-1 段音频历史。规模：train
12,500 行 / dev 355 行（比 62,707 次调用少一个量级）；超过 4096-frame
上限的 row 按句群分段合成。

**v2 已拍板决策（luojiaxuan, 2026-08-04）**：
1. **固定音色**，不做英语说话人音色克隆。固定 ref 用 MOSS 无条件生成的
   中性中文段落（9.44s, 24kHz）：
   `RUN_ROOT/fixed_ref/fixed_zh_ref.wav`，生成与训练全程复用同一条。
2. **多 turn SFT**：一个 row 一条记录、N 个 assistant turn（各配 FA 切片
   codes），生成第 k 段时能看到前 k-1 段音频历史，与流式推理一致。
   `dataset.py` 原生支持（`ref_audio_codes` optional，多 turn 逐 turn 设
   label）。

## 2026-08-04 v2 pipeline 实施记录

v2 run root：`/data/S2S_omni_runs/moss_tts_infinisst_v2_20260804`

脚本（commit `ea0f157`）：

- `scripts/build_moss_v2_row_requests.py`：segment manifest 按 row 分组，
  纯标点 segment 并入邻段，>400 字符的 row 按句末标点分组（本数据集全部
  单组）。产出：train 12,500 行 / 62,375 段，dev 354 行 / 798 段。行长
  分布：train max 197 可发声字符（约 55s 音频）、中位 80；dev max 86。
- `scripts/run_moss_v2_serving_4way.sh`：GPU0-3 各一个 `sgl-omni serve`，
  端口 `48731,49157,52391,54863`，`--allowed-local-media-path RUN_ROOT`。
- `scripts/generate_moss_realtime_long_targets.py`：每 row 全文一次合成
  （固定 ref），runaway 预算 `max(15s, 可发声字符×0.6s)`、重试 2 次，
  按 row_id resume。
- `scripts/align_slice_moss_v2.py`：zh CTC forced alignment
  （`jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` +
  `torchaudio.functional.forced_align`）→ 相邻 segment span 中点为切点 →
  MOSS-Audio-Tokenizer 整行编码一次（实测 fps 恰好 12.5）→ codes 帧级
  切片 → 直接产出多 turn prepared record（跳过 `prepare_data.py`）。
  数字/拉丁字符不在 aligner 词表则跳过，整段无对齐字符按字符数比例插值；
  coverage < 0.5 的 row 进 audit 排除。
- 3 行 dev smoke 全绿：coverage 0.966-1.0，边界语言学上合理，record 经
  `MossTTSRealtimeSFTDataset` 打包通过（`input_ids (T,17)`；32 codebook
  被 dataset 裁到模型的 16 通道，与 v1 prepare 行为一致）。

全量生成经历三次拓扑调整（hyper00 4-way -> hyper00 4 + hyper01 4 ->
按 luojiaxuan 指示 hyper00 2 + hyper01 6），最终 `2026-08-04T22:0xZ`
全部完成：**12,852/12,854 accepted，2 条合法 runaway reject，零跨机
重复**。hyper01 环境为复制 hyper00 容器（`jaxanluo/sglang-omni:dev` +
editable install PR1192 `caa77bf6` + 按 hyper00 pip freeze 补齐 62 个
缺失包）。

**事故记录（重要教训）**：第一次 rebalance 用 `kill $(cat pidfile)` 只杀了
launcher 子 shell，4 个 stage1 python worker 成为孤儿继续运行；随后停掉
GPU2/3 serving 时，孤儿 shard2/3 将 5,159 行烧成 `Connection refused`
伪 reject（隔离于 `raw/quarantine_bogus_rejects_20260804/`），且当时的
重分片把 rejected 当作 done 排除，这批行一度无人认领。修复：进程树 kill
（先 TERM 后 KILL 遍历子进程）、以 accepted+合法 reject 重算未完成集、
完成判定改为 worker 全部退出 + row_id 去重覆盖校验。**此后所有 worker
停止必须树杀，且 reject 文件在重分片前必须人工过目。**

align/slice 分两波流水线执行（wav 在哪台机器就在哪台对齐）：hyper00
wave1 4,147 行（2 卡后按指示扩到 5 卡），hyper01 wave2 8,705 行
（6 卡）。结果：train 12,498 行 / 62,373 turns，dev 354 行 / 798
turns，**audit 零排除**，coverage 中位数 1.0（train p5=0.97 min=0.75）。
wave2 prepared JSONL（493MB）经 Mac 中转合并回 hyper00。

**v2 SFT 完成（2026-08-04T22:15Z）**：4x H200 DDP bf16 sdpa，1 epoch
= 782 steps（多 turn 记录，总监督 token 与 v1 3,893 步相当），loss
4.29 -> ~3.50，~5 分钟。checkpoint：
`RUN_ROOT/checkpoints/moss_tts_realtime_infinisst_v2_multiturn/checkpoint-epoch-0`
上传 HF private repo：
`gavinlaw/moss-tts-realtime-infinisst-en-zh-v2-multiturn`。

## 完整链路 demo（2026-08-04）

链路：自构造 50.6s 英文播客音频（10 句，MOSS 合成，逐句 offset 已知）
-> Taurus 上 InfiniSST no-RAG baseline
（`/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_origin-bsz4` =
HF `gavinlaw/infinisst-no-tmsft-origin-bsz4-zh`）经 RASST
`20260524__batched_vllm_rag_eval.sh`（`DISABLE_RAG_OVERRIDE=1`，vllm
TP4，0.96s chunk）流式推理，53 chunk 全部产出中文增量 -> 每段以对应
0.96s 源切片为 ref 调 v1 finetuned TTS -> 拼接 70.1s 中文。

- demo 资产：hyper00
  `RUN_ROOT/demo/full_chain/`（源音频、runtime_chunks.jsonl、逐段
  ref/tts wav、拼接结果、segment_table.json）；Taurus 输入与 runtime 在
  `/mnt/gemini/data1/jiaxuanluo/moss_tts_full_chain_demo_20260804/`。
- 观察：译文连贯正确；本次 53 段无 runaway（但 dev_r000002 对照 demo
  中 v1 对 6 字符 fragment 曾跑出 14.7s，重试 0.88/4.0/6.0s，证明
  runaway 随机且 v1 未根治）；源 50.6s -> 目标 70.1s（+38% backlog）。
- v1/base/v2 式四路对照音频（dev_r000002）与本 demo 音频均已交付
  luojiaxuan 试听。
- v1 serving: hyper00 GPU5 port 45001；v2 serving: hyper00 GPU6 port
  47111（单发 `/v1/audio/speech` 是无历史推理，与 v2 多 turn 训练形态
  有 gap；真正的多 turn 增量推理接入是下一步）。
- v2 checkpoint 已上传 HF：
  `gavinlaw/moss-tts-realtime-infinisst-en-zh-v2-multiturn`
  （commit `0c478bb85680b00ca0124d0c89a934eabee10561`）。

## 多 turn 增量推理与 dev 正式评测（2026-08-04/05）

多 turn 增量推理已接入：`scripts/moss_multiturn_infer.py`（commit
`854d214`）。复用上游 `MossTTSRealtimeStreamingSession`：assistant-only
turn 通过自定义 `input_ids` 实现（turn header 与 finetuning packer 逐
字节一致），固定音色 voice prompt 进 ensemble，KV cache 跨 turn 保留，
`prefill_text_len=delay_tokens_len` 保持轮内 text/audio delay 交错与
训练一致，逐 turn runaway 帧预算保护。**训推一致性说明**：训练侧的
"改动"就是 v2 多 turn 记录格式本身，两侧遵循同一上游序列约定；残余
gap 是 exposure bias（标准教师强制差距）、会话长度外推（训练行 ≤~13
turns）、历史音频音源（训练=base 整行合成切片 vs 推理=v2 自产）。

dev 354 行正式评测（`scripts/eval_moss_dev_systems.py`，commit
`34de262`；whisper large-v3 zh ASR，参考=InfiniSST segment 文本拼接，
BLEU tokenize=zh；duration ratio = target 音频/源音频时长）：

| 指标 | v1 per-segment | v2 multi-turn |
| --- | ---: | ---: |
| rows scored / failed | 353 / 1 | 352 / 2 |
| BLEU(zh) | 72.31 | **74.96** |
| chrF | 65.95 | **69.73** |
| CER mean / median | 0.1646 / 0.0919 | **0.1341** / 0.0892 |
| duration ratio mean | 1.156 | **0.996** |
| duration ratio median | 1.050 | **0.883** |
| duration ratio p90 | 1.667 | **1.333** |
| runaway turns | 4 / 796 | **0 / 774** |

结论：v2 + 多 turn 推理把 dev 平均时长比压到 ~1.0（v1 为 1.16、p90
1.67），**774 turns 零 runaway**，同时可懂度（BLEU/chrF/CER）反而更好
——连贯性没有以保真度为代价。smoke 例证：dev_r000002 v1 逐段 30.2s ->
v2 多 turn 13.4s（整行合成参考 12.2s）。

session-reset 对策已验证：demo 53 轮按每 11 轮重置 session（5 个会话，
`demo_rows_sessionreset.jsonl`），全部完成零 runaway，总时长 58.4s
（源 50.6s，时长比 1.15；对比 v1 逐段 70.1s/1.39、v2 单发 71.8s/1.42），
输出 `demo/full_chain/demo_full_chain_v2_multiturn.wav`。

已知限制：demo 的 53-turn 单会话在 turn 47 两次 runaway（floor 8s 与
15s 各一次）——远超训练行长（≤~13 turns）的分布外失稳，对应部署对策
是每 ~10-12 turns 重置 session，数据侧对策是 v3 拼接长会话训练。

评测产物：`RUN_ROOT/eval_v1_dev/`、`RUN_ROOT/eval_v2_multiturn/`
（per-row scored JSONL + report JSON + 全部 wav）。

- v2 单发（固定中文 ref、无历史）复跑同 53 段：71.8s，无 runaway，与
  v1 70.1s 相近——**单发推理吃不到多 turn 训练收益**，时长/韵律收益需
  多 turn 增量推理接入后评估。三段音频（源/v1/v2）均已交付试听。

## 当前状态

更新时间：`2026-08-04T17:13:00Z`（后续进展见上两节，以上两节为准）

当前 Git 分支：

```text
moss-tts-infinisst
```

当前 full run root：

```text
/data/S2S_omni_runs/moss_tts_infinisst_20260804_0939
```

已完成：

- 数据字段和 MOSS finetuning 格式已确认。
- `source_text` 不是硬依赖：MOSS-Realtime serving 可以只用 `ref_audio`。
- Taurus 已生成 full tar.zst package，本机和 hyper00 校验通过。
- HF private dataset 已上传，commit 为
  `c54868f00916e26c7b3893149f2ce43aa13f9632`。
- hyper00 已下载并解压 source wavs：
  - train source wavs: `62,707`
  - dev source wavs: `801`
- hyper00 smoke 已通过：
  - `/v1/models` ready。
  - 3 条 dev target generation 全 accepted。
  - `prepare_data.py` 3 条 smoke 成功，输出带 `audio_codes` 的 JSONL。
  - `sft.py` 1-step smoke 成功，loss `3.2439`，checkpoint 写出。
- full target generation 已覆盖完整 train/dev manifest。当前计数：
  - train: `accepted=62381`, `rejected=326`, `covered=62707/62707`,
    `missing_or_pending=0`, `bad_reject_count=0`
  - dev: `accepted=796`, `rejected=5`, `covered=801/801`,
    `missing_or_pending=0`, `bad_reject_count=2`
- full target generation shard 进程已结束：
  - shard0: `accepted=15593`, `rejected=84`
  - shard1: `accepted=15597`, `rejected=80`
  - shard2: `accepted=15590`, `rejected=87`
  - shard3: `accepted=15601`, `rejected=75`

尚未完成：

- `prepare_data.py` 和 full SFT 尚未开始；当前 `02_prepare_train.log`,
  `02_prepare_dev.log`, `03_train.log` 仍不存在。
- `prepared_jsonl_count=0`。
- 当前 checkpoint 只有 smoke checkpoint，不是 full training 结果：

```text
/data/S2S_omni_runs/moss_tts_infinisst_20260804_0939/checkpoints/smoke_1step/checkpoint-epoch-0/model.safetensors
```

当前风险点：

- dev 有 `bad_reject_count=2`。后半段 supervisor 在进入
  `prepare_data.py` 前会校验 accepted + rejected 是否覆盖完整 manifest，
  且 rejected 是否只包含无发声字符。这个 dev bad reject 很可能会让
  post-generation stage 卡住或退出。
- `target_wavs/dev` 文件数曾观察到 `797`，而 reporter accepted 是 `796`；
  以 raw shard JSONL 和 reporter 结果为准，不要只看目录文件数。
- hyper00 container 的 `/data` 挂载是 `/data04/jaxan`，当前可用空间充足。
  `/data02` 已满但不应影响这个 run。

## 交接给下一个 Claude session

（2026-08-04 更新）dev bad rejects 已修复、overlong 过滤已加入、prepare + full
SFT supervisor 已于 `2026-08-04T17:41Z` 左右重启（容器内 PID 见
`pids/prepare_train_supervisor.pid`，环境 `CUDA_DEVICES=0,1,2,3 NUM_EPOCHS=1
MAX_TARGET_DURATION_S=30`）。下一 session 接手时先跑 status 命令确认阶段：
若 `logs/03_train.log` 存在且
`checkpoints/moss_tts_realtime_infinisst_train/checkpoint-*` 已产出，则 full
SFT 完成，下一步是把 checkpoint 上传 HF（模型参数以 HF 为 source of truth）
并做 streaming segment 推理评测；若 supervisor 已死且无 checkpoint，先看
`logs/99_prepare_train_supervisor.log`（attempt1/attempt2 历史保存在同目录
`*.attempt1.log` / `*.attempt2.log`：attempt1 是 dev bad-reject validation
失败，attempt2 是 runaway target 导致的 prepare OOM）。

以下为历史交接内容（bad reject 检查已完成，不需重做）。

必须先看的状态命令：

```bash
ssh hyper00
docker exec -it sglang-omni-jaxan bash

python /data/S2S_omni/scripts/report_moss_tts_run_status.py \
  --run-root /data/S2S_omni_runs/moss_tts_infinisst_20260804_0939 \
  --dataset-root /data/datasets/infinisst-moss-tts-en-zh-segments-v1
```

检查 post-generation supervisor：

```bash
tail -120 /data/S2S_omni_runs/moss_tts_infinisst_20260804_0939/logs/99_prepare_train_supervisor.log
```

检查 dev bad rejects：

```bash
for f in /data/S2S_omni_runs/moss_tts_infinisst_20260804_0939/raw/dev_moss_rejected_shard*.jsonl; do
  echo "===== $f"
  tail -20 "$f"
done
```

如果 bad rejects 确认只是中文标点、空白、不可发声字符，最小修复是放宽
validation 或将这些样本标成合法 reject，然后只重跑 post-generation stage：

```bash
RUN_ROOT=/data/S2S_omni_runs/moss_tts_infinisst_20260804_0939 \
DATASET_ROOT=/data/datasets/infinisst-moss-tts-en-zh-segments-v1 \
S2S_OMNI_ROOT=/data/S2S_omni \
MOSS_TTS_ROOT=/data/MOSS-TTS \
CUDA_DEVICES=0,1,2,3 \
NUM_EPOCHS=1 \
nohup bash /data/S2S_omni/scripts/run_moss_prepare_and_sft_after_generation.sh \
  > /data/S2S_omni_runs/moss_tts_infinisst_20260804_0939/logs/99_prepare_train_supervisor.log 2>&1 &
```

如果 bad rejects 是有实际可发声内容的中文 segment，则不要直接忽略；应先
定位对应 request，单独 regenerate 那几条 dev target wav/raw JSONL，再跑
post-generation stage。

不要做：

- 不要重跑 `run_moss_target_generation_4way.sh` 覆盖全量 62,707 条 train
  target generation。
- 不要把 smoke checkpoint 当 full checkpoint。
- 不要把 target wavs 或 checkpoints 提交进 Git；大 artifact 应上传 HF。
- 不要使用超过 4 张 hyper00 GPU，除非用户明确授权。


## ACL6060 benchmark 与 serving PR 计划（2026-08-05 进行中）

luojiaxuan 指示：(1) 滑动窗口多 turn session 包进 sglang-omni PR 1192；
(2) 用 ACL6060 canonical benchmark（5 talks，与 GPT/Gemini 同表）测
InfiniSST + v2 TTS cascade；(3) 多用 GPU。

**ACL 推理阵地转移**：Taurus 8 卡被 zili 的
`ucsb_qwen_gpu_placeholder.py`（8x44GB、5s 采样全程 0%）占住，符合回收
授权但无 sudo 杀不掉他人裸进程（授权与 Unix 权限冲突，已记录在
`/mnt/gemini/data1/jiaxuanluo/moss_tts_full_chain_demo_20260804/reclaim_log.txt`）。
改在 hyper01 跑：H200 单卡装下 30B（免 TP）。准备（进行中）：
- RASST eval 代码（2.2MB）已拷到 hyper01 `/data/RASST_eval/code/rasst`
  （源：Taurus `/mnt/data2/jiaxuanluo/RASST`，remote `LeiLiLab/RASST`）。
- 容器内独立 venv `/data/venvs/vllm_eval`：vllm==0.13.0（对齐 Taurus
  spaCyEnv），不碰 sglang 主环境。
- HF 下载：`gavinlaw/infinisst-no-tmsft-origin-bsz4-zh`（~60GB）+
  `gavinlaw/rasst-main-result-data` 的 acl6060 音频与 acl_zh inputs。

ACL 跑法（与 demo 相同 launcher，改指 acl_zh 全量 inputs）：
`20260524__batched_vllm_rag_eval.sh` + `DISABLE_RAG_OVERRIDE=1`、
`LMS_OVERRIDE=1`、TP1/单卡或 TP2、`SKIP_OFFLINE_EVAL_OVERRIDE=1`；
runtime chunks -> v2 多 turn TTS（滑动窗口）-> 与
`projects/acl6060_s2s_metrics_seed/artifacts/acl6060_full_table.tsv` 的
GPT/Gemini En-Zh 1x 行对比。质量分先出（SEGALE BLEU / XCOMET-XL，脚本
`scripts/run_acl6060_metric_pipeline.py` 系），LongYAAL/Ending Offset 需
v2 playout 协议的 arrival 时间线，第二步再接。

**serving PR 滑动窗口设计**（sglang-omni PR 1192，branch
`luojiaxuan/moss-tts-realtime`，实现待做）：
- API：`/v1/audio/speech` 增加可选 `session_id` + `session_window`
  （默认 ~8 turns）。带 session_id 的请求在 pipeline 侧维护
  per-session 历史（每 turn 的 text + 生成的 audio codes）。
- 请求构造：ensemble(ref codes) + 最近 W-1 turns 的
  `<|im_start|>assistant\n text ⊕ codes <|im_end|>\n` + 新 turn
  header，一次 prefill 后自回归生成（与
  `scripts/moss_multiturn_infer.py` 的布局一致，滑动窗口=重建 prompt，
  无需无限 KV）。
- 入口文件：`sglang_omni/models/moss_tts/request_builders.py`
  （`_build_processor_message` / `MossTTSState`）+ pipeline 的
  session store；`serve/test_speech_protocol.py` 加用例。
- 效率动机：sglang 引擎带 cuda graph，逐帧生成远快于脚本级
  transformers 推理（dev 实测 ~38s/turn 太慢，ACL 5 talks ~4k chunks
  必须走优化引擎或大幅并行）。


## v1 弃用决定（luojiaxuan, 2026-08-05）

v1（per-segment 自蒸馏 + source-chunk ref）在当前链路**宣告死刑**：dev
评测全面落后（时长比 1.156 vs 0.996、runaway 4 vs 0、BLEU/chrF/CER 均
劣于 v2），且其监督形态无法提供跨 chunk 连续性。此后主打 **v2 多 turn**
（固定音色 + 滑动窗口 session）。v1 checkpoint 仅作历史 baseline 保留在
HF `gavinlaw/moss-tts-realtime-infinisst-en-zh`，hyper00 GPU5 的 v1
serving 已停止，ACL benchmark 只跑 v2 臂。


## ACL benchmark 架构修正（luojiaxuan, 2026-08-05）

放弃独立 vllm venv 方案（两台机器的安装已停止清理）。全 serving 化：

1. **S2T**：`sgl-omni serve` 直接起 InfiniSST checkpoint
   （`gavinlaw/infinisst-no-tmsft-origin-bsz4-zh`，Qwen3-Omni-30B，单
   H200）。RASST 代码只作协议参考（0.96s chunk、cache 16/8、增量续写
   prompt，见 hyper00 `/data/RASST_eval/code/rasst/eval/src/
   batched_vllm_rag_eval.py`），另写轻量 streaming client。
2. **TTS**：v2 多 turn 走 sglang-omni PR 1192 的滑动窗口 session（先
   实现任务 #12，benchmark 直接消费）。
3. **评测**：`kit-lecture-translator` 分支的 portable 入口
   （`run_acl6060_full_table.sh` / `run_acl6060_metric_pipeline.py`，
   remote HEAD `881ee3f`，18 个 GPT/Gemini cell 审计齐全），本地
   checkout `/Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/
   S2S_omni-acl6060-sot`。主表对比行：`acl6060_full_table.tsv` 的
   En-Zh 1x GPT/Gemini。

执行顺序：serving PR 滑动窗口 session -> InfiniSST serving + client ->
5 talks cascade -> 指标流水线。InfiniSST checkpoint 与 ACL 数据已在
hyper00/hyper01 双机 HF cache 就位。


## serving PR 滑动窗口实现地图（2026-08-05，实现中）

benchmark 配置确认（luojiaxuan）：no-RAG，chunk size 测 0.96s 和 1.92s
两档。

sglang-omni PR 现状：`WS /v1/audio/speech/stream`
（`sglang_omni/serve/speech_ws.py`, `SpeechWebSocketSession`）是传输级
session——`_handle_input_text` -> `_pop_complete_segments` ->
`_generate_sentence`，每句作为独立单发请求进 pipeline，**无模型级
多 turn 历史**。

实现改动点（3 处）：
1. `serve/speech_ws.py`：session 内维护滑动窗口历史
   `[(text, audio_codes), ...]`（最近 W-1 turns，`session.config` 增加
   `multiturn: true` / `session_window`，默认 8）；每句生成完把
   (text, codes) 入队。生成的 codes 需从 pipeline 结果带回（tts_engine
   stage 在 vocoder 前就有 codes，加 `return_codes` 透传）。
2. `models/moss_tts/payload_types.py` + `request_builders.py`：
   `MossTTSState` 增加 `history` 字段；prompt 构造 = ensemble(ref) +
   最近 W-1 turns 的 `<|im_start|>assistant\n text ⊕ codes
   <|im_end|>\n` + 新 turn header + 新文本（布局与
   `scripts/moss_multiturn_infer.py` / finetuning packer 逐 token 一致）。
3. `tests/unit_test/serve/test_speech_protocol.py`：multiturn 会话用例。

S2T client（步骤 2）：对 sgl-omni serve 的 InfiniSST Qwen3-Omni 起
chunked streaming client，协议参数抄 RASST
`batched_vllm_rag_eval.py`（no-RAG prompt policy、cache 16/8 chunks、
max_new_tokens 40 lm_scaled、temperature 0.6/top_p 0.95/top_k 20），
chunk 0.96s 与 1.92s 两档各跑 5 talks。


## serving 滑动窗口 session 现状（2026-08-05）

已实现并推送（sglang-omni `luojiaxuan/moss-tts-realtime`，commits
`fc3f599` 增加 `/v1/audio/speech` history 字段 + prompt 渲染，
`c962dbc`/`8780781` 单帧与零帧历史轮规范化，`79260d8` 零长度轮布局）。
机械层通过：13 轮链式请求全程 200，prefix cache 生效。

**质量未达标（experimental）**：13 轮 session 总时长 9.5s（脚本参考
13.4s），5 轮零帧 deferred，whisper CER 0.669（脚本路径 dev 中位
0.089），语音内容损坏。疑因：serving 引擎当前轮 BOS 放在
`min(len(text),12)` 处、model_runner 的 delay 流式机制与训练 packer 的
短文本分支布局不一致；零帧历史轮也是训练分布外。**待调试**：diff
serving `model_runner` 的 delay 处理 vs 上游
`streaming_mossttsrealtime` session，短文本轮对齐训练布局。

**决策**：ACL benchmark 的 TTS 采用已验证的
`scripts/moss_multiturn_infer.py` 脚本路径（dev CER 0.089、时长比
0.996）；serving history 特性留 PR 继续修，不阻塞 benchmark。


## ACL6060 canonical 首轮结果与诊断（2026-08-05）

SEGALE 流水线已在 hyper00 打通（venv `/data/venvs/segale_eval2`，坑：
vecalign 为 Speech-to-Speech-Latency 仓库内置需 `.pth`+Cython+bin shim、
setuptools<81 保 pkg_resources、COMET 需补 click/rich/typer、spaCy 需
zh_core_web_sm）。对齐与打分均为 canonical 协议（pinned `d0041438`，
spacy+LaBSE+vecalign max_size=8，BLEU tokenize=zh 保留标点 null 记零，
XCOMET-XL reference-free source+hypothesis）。

| 系统 (En-Zh 1x) | BLEU | XCOMET-XL | null 率 |
| --- | ---: | ---: | ---: |
| GPT-realtime (主表) | 32.70 | 0.628 | ~8% |
| Gemini 3.5-live (主表) | 40.39 | 0.586 | ~8% |
| KIT (主表) | 34.76 | 0.588 | ~8% |
| cascade chunk=1.92s | 23.96 | 0.291 | 28% (under 72/320) |
| cascade chunk=0.96s | 6.86(无效) | 0.140 | 48% (under 130/284) |

诊断：
- **0.96 档数字无效**：sglang client 在小增量+16/8 修剪下触发重复翻译
  退化（talk 268 假设 9,164 字符 vs 参考 3,532）。1.92 档文本量与参考
  一致，确认是 client 修剪/续译语义与 RASST 原实现的偏差被小 chunk 放大。
  另 `extra_body.top_k` 对 server 无效（OpenAI 客户端库概念）。
- **1.92 档为下界**：BLEU 23.96 主要损失是 22.5% under-translation
  （7 个 TTS runaway session 丢音频 + session 末尾 deferred 音频未 flush
  + ASR 损耗）。
- 产物：`RUN_ROOT/acl_bench/rundirs/acl6060_live_enzh_cascade_mossv2_
  chunk{096,192}_speed1/`（instances.log、segale_alignment、
  bleu_summary.json、xcomet_summary.json、xcomet_segments.jsonl）。

修复计划：0.96 client 与 Taurus RASST runtime 逐 prompt 对拍；TTS 补
runaway 重试与末轮 flush；重跑两档后更新本表。latency 列（LongYAAL/
Ending Offset）仍为 uniform proxy timing，不可比，待 speech-playout v2
协议接入。


## v2.1 重复鲁棒性增强（2026-08-05）

滑窗推理暴露的音频循环自强化问题（详见上节），四个推理侧补丁臂
（A: repetition_window 375/1.15、B: n-gram 检测清窗、C: 检测+重生成、
D: 音频秒/字符比检测清窗）在 probe268（talk 268 前 100 轮）上 CER
0.39-0.69，均无法根治——n-gram/token 级手段失效的根因是循环音频在
RVQ codes 上不逐字复现。luojiaxuan 决策转训练侧。

**v2.1 方案**：`scripts/augment_moss_v2_repetition.py` 对 ~20% 训练行
在中间某轮 audio codes 内拼接复制 8-24 帧短语（模拟跳针历史），该轮标
`context_only` 不进 loss（finetuning `dataset.py` 已打局部补丁：
`is_assistant and not turn.get("context_only")`），后续轮保持干净监督。
12,498 原行 + 2,205 增强行 = 14,703 条，1 epoch 919 步重训（超参同 v2）。
checkpoint：`RUN_ROOT/checkpoints/moss_tts_realtime_infinisst_v21_repaug/`。

probe268 对照（裸滑窗，无推理补丁）：
- v2: CER 0.52-0.69，中段病态锁死循环
- **v2.1: CER 0.396，锁死循环消失**，仅剩局部小结巴

（v2.1 + ratio 检测组合探针进行中；MOSS-TTS 本地补丁清单：
`modeling_mossttsrealtime_local.py` create_causal_mask 适配 transformers
5.6 签名、`dataset.py` context_only，均留 .bak。）


## ACL6060 benchmark 终局汇总（2026-08-05）

| 系统 (En-Zh 1x, canonical SEGALE BLEU / XCOMET-XL) | BLEU | XCOMET | 备注 |
| --- | ---: | ---: | --- |
| GPT-realtime (主表) | 32.70 | 0.628 | |
| Gemini 3.5-live (主表) | 40.39 | 0.586 | |
| KIT (主表) | 34.76 | 0.588 | |
| **cascade v2 + session-reset, chunk=1.92s** | **23.96** | **0.291** | **当前操作点** |
| cascade v2 + 滑窗 | 4.80 | 0.172 | 循环自强化 |
| cascade v2.1(repaug) + 滑窗 | 6.61 | 0.162 | 轻度增强不足 |
| cascade chunk=0.96s | 无效 | 无效 | client 修剪语义缺陷待修 |

结论与教训：
- v2.1 的 20% 单轮 context_only 污染在 100 轮探针上显著改善
  （CER 0.52-0.69 -> 0.396、锁死循环消失），但 300+ 轮整 talk 尺度仍
  退化（时长比膨胀至 ~1.25，BLEU 6.61）——**探针尺度会低估自强化循环，
  结论必须以整 talk 验证为准**。
- 推理侧四臂（长回看 token 惩罚 / n-gram 检测 / 检测重生成 / 秒字符比
  检测）均不能根治；RVQ codes 不逐字复现使精确匹配类手段失效。
- **当前对外可引用数字：v2 + session-reset(11 轮) chunk=1.92s，
  BLEU 23.96 / XCOMET 0.291**（含 22.5% under-translation，为 cascade
  下界；latency 列 timing 为 proxy 不可比）。

v3 计划（根治方向，按优先级）：
1. 训练数据拼接长会话（10-30+ 轮，覆盖整 talk 尺度的窗口滑动形态）；
2. 更强污染增强：fraction 0.5+、多轮污染、跨轮循环模式（不只轮内
   拼接）、污染位置含窗口边界；
3. scheduled sampling：训练历史混入模型自产 codes；
4. 0.96 InfiniSST client 与 RASST 原实现逐 prompt 对拍修复后补 0.96 档；
5. TTS runaway 行级中止改轮级跳过 + 自动重试（117 曾在倒数第 5 轮
   全行报废）。

产物：v2.1 checkpoint
`RUN_ROOT/checkpoints/moss_tts_realtime_infinisst_v21_repaug/`（未上传
HF——非操作点，留本地；如需消融可后补），全部 run dirs 与分数在
`RUN_ROOT/acl_bench/rundirs/`。


## v3 结果：cascade 追平商用系统档位（2026-08-05）

v3 训练（`scripts/build_moss_v3_dataset.py`，commit `fb45fdb`）：12,498
原行 + 1,987 条 20-35 轮长会话 + 6,355 条强污染副本（轮内跳针/跨轮
carry-over/整轮 replay，均 context_only）= 20,840 条，3xH200 1 epoch
1,390 步。推理侧同步落地 runaway 轮级跳过（滑窗模式下截断+清窗继续，
整 talk 不再因单轮报废）。

ACL6060 canonical 四象限终表（En-Zh，chunk=1.92s，SEGALE BLEU /
XCOMET-XL reference-free）：

| 系统 | BLEU | XCOMET | null 率 |
| --- | ---: | ---: | ---: |
| GPT-realtime (主表) | 32.70 | 0.628 | ~8% |
| Gemini 3.5-live (主表) | 40.39 | 0.586 | ~8% |
| KIT (主表) | 34.76 | 0.588 | ~8% |
| cascade v2 + reset | 23.96 | 0.291 | 28% |
| cascade v2 + 滑窗 | 4.80 | 0.172 | 34% |
| cascade v3 + 滑窗 | 16.41 | 0.303 | 22% |
| **cascade v3 + reset（新操作点）** | **34.69** | **0.554** | **6.8%** |

**v3 + session-reset：BLEU 34.69 超过 GPT-realtime（32.70）、贴平 KIT
（34.76），XCOMET 0.554 进入商用档位，null 率 6.8% 与主表系统持平，
160 session 零失败。** 长会话+强污染训练同时修复了段内质量与
under-translation；reset 模式则隔绝循环自强化——组合即当前最优。
滑窗模式在 v3 下也大幅回血（4.8 -> 16.4）但仍逊于 reset，彻底滑窗化
留给 v4（scheduled sampling / 更强长程训练）。

checkpoint：`gavinlaw/moss-tts-realtime-infinisst-en-zh-v3-longsess`
（上传中，commit 见 HF）；本地
`RUN_ROOT/checkpoints/moss_tts_realtime_infinisst_v3/`。run dir：
`acl_bench/rundirs/acl6060_live_enzh_cascade_mossv3_reset_chunk192_speed1/`。
仍未完成：0.96 档 client 对拍、latency 列 speech-playout 协议接入、
serving PR parity（v3 checkpoint 下需复验）。
