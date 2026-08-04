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

全量生成于 `2026-08-04T19:0xZ` 启动：4-way，速率 ~25 行/分钟，ETA ~8.5h。
监控：host 侧 `monitor_moss_20260804/check_v2_generation.sh` + Mac 侧
10 分钟轮询（shard 死亡/停滞/失败告警，小时级进度心跳，完成判定
accepted+rejected == 12,854）。

生成完成后的步骤：停 serving → `align_slice_moss_v2.py` 4 GPU 分片跑
train+dev → `sft.py --train-jsonl RUN_ROOT/prepared/train_v2_*.jsonl`
（sdpa，多 turn records）→ checkpoint 上传 HF。

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
