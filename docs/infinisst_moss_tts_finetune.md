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

该 launcher 会执行：

1. `scripts/generate_moss_realtime_targets.py`
2. `moss_tts_realtime/finetuning/prepare_data.py`
3. `moss_tts_realtime/finetuning/sft.py`

## 当前状态

- 数据字段和 MOSS finetuning 格式已确认。
- `source_text` 不是硬依赖：MOSS-Realtime serving 可以只用 `ref_audio`。
- Taurus 已生成 full tar.zst package，本机和 hyper00 校验通过。
- HF private dataset 已上传，commit 为 `c54868f00916e26c7b3893149f2ce43aa13f9632`。
- hyper00 已下载并解压 source wavs。
- hyper00 已启动 MOSS-TTS-Realtime serving smoke：
  - `/v1/models` ready。
  - 3 条 dev target generation 全 accepted。
  - `prepare_data.py` 3 条 smoke 成功，输出带 `audio_codes` 的 JSONL。
  - `sft.py` 1-step smoke 成功，loss `3.2439`，checkpoint 写出。
- target generation 中纯标点 segment 例如 `，` 会被本地 reject，不参与 TTS
  训练。这类 chunk 没有可发声内容，不能强制 MOSS 生成 wav。
- 当前 full target generation 正在 hyper00 运行，run root：

```text
/data/S2S_omni_runs/moss_tts_infinisst_20260804_0939
```

- 需要继续执行：等待 full target wav 生成完成，合并 raw shards，运行
  `prepare_data.py`，再运行 4-GPU `sft.py`。
