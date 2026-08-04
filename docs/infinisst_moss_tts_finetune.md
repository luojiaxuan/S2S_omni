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

## 当前状态

更新时间：`2026-08-04T17:13:00Z`

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

下一 session 首要任务不是重跑 target generation，而是检查后半段 supervisor
为什么没有进入 prepare/SFT。full target wav 生成已经覆盖完整 manifest。

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
