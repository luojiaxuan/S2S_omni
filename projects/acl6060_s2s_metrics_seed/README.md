# ACL6060 Seed S2S Metrics Script

This project bundle tracks the Seed / ByteDance AST speech-to-speech evaluation
script provided under:

```text
/Users/luojiaxuan/Downloads/seed
```

The imported source lives in:

```text
vendor/seed/
```

## Contents

- `vendor/seed/generate.py`: streams wav audio to the AST S2S WebSocket API and
  writes target wav, timeline JSON, and transcript text.
- `vendor/seed/protos/`: protobuf definitions and generation helper.
- `vendor/seed/python_protogen/`: generated Python protobuf files required by
  `generate.py`.

Runtime cache files such as `__pycache__` and `*.pyc` are intentionally not
tracked.

## Credential Handling

The original downloaded script contained hard-coded AST credentials. Those have
been removed before committing to this public repository.

Pass credentials explicitly at runtime:

```bash
python projects/acl6060_s2s_metrics_seed/vendor/seed/generate.py \
  input.wav \
  --out-dir output/seed/en_ja \
  --src-lang en \
  --tgt-lang ja \
  --api-key '<AST_API_KEY>'
```

For legacy two-part auth:

```bash
python projects/acl6060_s2s_metrics_seed/vendor/seed/generate.py \
  input.wav \
  --out-dir output/seed/en_ja \
  --src-lang en \
  --tgt-lang ja \
  --app-key '<AST_APP_KEY>' \
  --access-key '<AST_ACCESS_KEY>'
```

## Dependencies

```bash
pip install soundfile numpy websockets "protobuf>=6.31" grpcio grpcio-tools
```

If protobuf definitions need to be regenerated:

```bash
cd projects/acl6060_s2s_metrics_seed/vendor/seed/protos
./build_python.sh
```

## Outputs

For each input wav, the script writes:

- `<stem>.wav`: translated target speech with silence gaps rendered on the live
  timeline.
- `<stem>_timeline.json`: receive/playback timing data for target chunks.
- `<stem>.txt`: subtitle transcript emitted by the backend.

## ACL6060 GPT/Gemini Streaming Reproduction

Use `run_acl6060_live_stream_eval.py` for the ACL6060 main-result protocol. It
uses the HF/RASST release layout with 5 full wav inputs, streams each wav into
OpenAI Realtime or Gemini Live in fixed audio chunks, and writes a 5-row
RASST/SimulEval-style `instances.log` with per-target-unit delays.

The release data is hosted at:

```text
https://huggingface.co/datasets/gavinlaw/rasst-main-result-data
```

Download only the ACL6060 EN->ZH subset and run a short protocol smoke:

```bash
python projects/acl6060_s2s_metrics_seed/run_acl6060_live_stream_eval.py \
  --dataset-root /tmp/rasst_main_result_data \
  --output-dir /tmp/acl6060_stream_openai_smoke \
  --provider openai \
  --api-key-file /path/to/openai.key \
  --download-hf \
  --chunk-ms 960 \
  --limit 1 \
  --max-audio-seconds 30 \
  --no-resume
```

Run the full OpenAI or Gemini stream by removing `--limit` and
`--max-audio-seconds`. The 5 ACL6060 full wavs are about 57.4 minutes total, so
the default paced live run takes about an hour. `--no-pace` sends chunks as fast
as the WebSocket accepts them; that is useful for protocol debugging, but it is
not a live wall-clock measurement. For Gemini full runs, pass
`--max-session-input-s 480` so long talks are split into service-sized sessions.
需要保留 API 返回的 target speech 时显式传 `--save-output-audio`；runner
会把 OpenAI `session.output_audio.delta` 或 Gemini
`serverContent.modelTurn.parts[].inlineData` 写为 24kHz mono WAV，并在
`responses.jsonl` 中记录路径、字节数和时长。默认不开启，正式 full-table
既有行为不变。

For the FLORAS-dashboard-style sweep over providers, chunk sizes, and input
speed factors, use:

```bash
scripts/run_acl6060_live_compare.sh \
  --providers openai,gemini \
  --chunks 960,1920 \
  --speeds 1,1.5
```

For a short smoke before an expensive full sweep:

```bash
scripts/run_acl6060_live_compare.sh \
  --providers openai,gemini \
  --chunks 1920 \
  --speeds 1.5 \
  --limit 1 \
  --max-audio-seconds 6 \
  --no-score \
  --no-resume \
  --output-base /tmp/acl6060_live_compare_smoke_fixed \
  --gemini-receive-timeout-s 30 \
  --gemini-post-send-idle-s 3
```

Score a completed `instances.log` in a RASST eval environment:

```bash
PATH=/mnt/taurus/home/jiaxuanluo/mwerSegmenter:$PATH \
/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python \
  /mnt/data2/jiaxuanluo/RASST/code/rasst/eval/offline_sst_eval/offline_streamlaal_eval.py \
  --mode acl6060 \
  --instances-log /path/to/stream_run/instances.log \
  --lang-code zh \
  --ref-file /tmp/rasst_main_result_data/main_result/inputs/acl_zh/ref.txt \
  --source-file /tmp/rasst_main_result_data/main_result/inputs/acl_zh/source_text.txt \
  --audio-yaml /tmp/rasst_main_result_data/main_result/inputs/acl_zh/audio.yaml \
  --glossary-acl6060 /mnt/data2/jiaxuanluo/RASST/data/glossaries/acl6060_tagged_gt_raw_min_norm2.json \
  --fbk-fairseq-root /mnt/taurus/home/jiaxuanluo/FBK-fairseq \
  --python-bin /mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python \
  --output-tsv /path/to/stream_run/eval_results.tsv \
  --output-log /path/to/stream_run/eval_results.log
```

Validated locally on 2026-07-04:

- HF subset download to `/tmp/rasst_main_result_data` was 107 MB.
- `source.list` and `target.list` contain 5 full-wav rows.
- `ref.txt` and `audio.yaml` contain 468 sentence rows for scorer
  resegmentation.
- A dry-run generated `/tmp/acl6060_stream_dry_run/instances.log` with
  `source[0]` pointing at `2022.acl-long.268.wav`, not a segmented sentence wav.
- Real 15-second API smoke tests succeeded for both providers:
  `/tmp/acl6060_stream_openai_smoke_real_fixed` and
  `/tmp/acl6060_stream_gemini_smoke_real`. Both produced Chinese target
  transcript deltas and 0 API errors.
- The compare script smoke for `chunk_ms=1920`, `speed_factor=1.5`, first
  6 seconds, succeeded for both providers with `source_length≈4009ms`, proving
  the speed factor is applied before streaming.

Do not pass API keys through environment variables. Use a local key file and do
not commit it.

### 2026-07-04 ACL6060 EN->ZH Live Sweep

These rows use the corrected 5-full-wav streaming input protocol, HF/RASST
release data, and RASST `offline_streamlaal_eval.py` scoring with
`lang_code=zh`. The sweep covers `chunk_ms=960,1920` and
`speed_factor=1.0,1.5`.

| provider | chunk_ms | speed | BLEU | masked BLEU | StreamLAAL | StreamLAAL_CA | TERM_ACC | TERM_ADOPTION | API errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenAI `gpt-realtime-translate` | 960 | 1.0 | 35.600 | 29.701 | 4403.721 | 4414.628 | 0.7404 | 0.4767 | 0 |
| OpenAI `gpt-realtime-translate` | 960 | 1.5 | 33.138 | 27.616 | -2649.876 | -2651.592 | 0.6764 | 0.4367 | 0 |
| OpenAI `gpt-realtime-translate` | 1920 | 1.0 | 35.104 | 29.484 | 4695.663 | 4690.100 | 0.7090 | 0.5667 | 0 |
| OpenAI `gpt-realtime-translate` | 1920 | 1.5 | 32.554 | 27.759 | -2619.640 | -2624.534 | 0.6427 | 0.4867 | 0 |
| Gemini `gemini-3.5-live-translate-preview` | 960 | 1.0 | 48.230 | 42.656 | 2428.819 | 2531.411 | 0.7472 | 0.5867 | 0 |
| Gemini `gemini-3.5-live-translate-preview` | 960 | 1.5 | 48.767 | 44.503 | -2879.656 | -2880.975 | 0.7584 | 0.4867 | 0 |
| Gemini `gemini-3.5-live-translate-preview` | 1920 | 1.0 | 47.488 | 41.642 | 34963.070 | 35119.306 | 0.7461 | 0.6667 | 0 |
| Gemini `gemini-3.5-live-translate-preview` | 1920 | 1.5 | 47.151 | 42.975 | -2850.625 | -2852.580 | 0.7258 | 0.4367 | 0 |

The `speed_factor=1.5` rows use compressed source-clock audio, so their
StreamLAAL values are not directly comparable with `speed_factor=1.0` rows.
Gemini had two transient WebSocket 1011 service-unavailable disconnects during
the sweep; both affected incomplete samples before row write and were rerun with
`--resume`, so all final rows have 0 API errors.

Tracked small artifacts:

```text
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_live_sweep_summary.tsv
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_live_sweep_summary.jsonl
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_live_openai_chunk960_speed1/
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_live_openai_chunk960_speed1p5/
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_live_openai_chunk1920_speed1/
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_live_openai_chunk1920_speed1p5/
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_live_gemini_chunk960_speed1/
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_live_gemini_chunk960_speed1p5/
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_live_gemini_chunk1920_speed1/
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_live_gemini_chunk1920_speed1p5/
```

Large local raw event/audio directories were not committed:

```text
/tmp/acl6060_live_sweep
/tmp/acl6060_stream_openai_full_chunk960
/tmp/acl6060_stream_gemini_full_chunk960
```

Taurus scorer work dirs:

```text
/mnt/data2/jiaxuanluo/tmp/s2s_omni_acl6060_live_sweep_20260704
/mnt/data2/jiaxuanluo/tmp/s2s_omni_acl6060_openai_chunk960_20260704
/mnt/data2/jiaxuanluo/tmp/s2s_omni_acl6060_gemini_chunk960_20260704
```

## ACL6060 GPT/Gemini Segmented Diagnostic

`run_acl6060_llm_audio_eval.py` runs an offline speech-to-text translation
baseline over the ACL6060 segmented source wavs. It writes SimulEval-compatible
`instances.log` records and scores them with the same SimulEval + sacreBLEU
path used by the old ACL6060 logs.

This is not the streaming-input reproduction. It makes 468 independent
sentence-level API calls and assigns each prediction to the end of the sentence
audio. Keep these artifacts only as a tokenizer/scorer diagnostic.

The ACL6060 dev data found during this pass lives on Taurus:

```text
taurus:/mnt/data/siqiouyang/datasets/acl6060
```

The old zh reference log used for scorer reproduction is:

```text
taurus:/mnt/data/siqiouyang/runs/infinisst_rag/offline/acl6060/zh
```

Example commands:

```bash
python projects/acl6060_s2s_metrics_seed/run_acl6060_llm_audio_eval.py \
  --dataset-root /path/to/acl6060 \
  --output-dir projects/acl6060_s2s_metrics_seed/artifacts/acl6060_dev_enzh_openai_gpt_audio_mini \
  --provider openai \
  --model gpt-audio-mini \
  --api-key-file /path/to/openai.key \
  --split dev \
  --target-lang zh \
  --deps-dir /path/to/simuleval_deps

python projects/acl6060_s2s_metrics_seed/run_acl6060_llm_audio_eval.py \
  --dataset-root /path/to/acl6060 \
  --output-dir projects/acl6060_s2s_metrics_seed/artifacts/acl6060_dev_enzh_gemini_audio \
  --provider gemini \
  --model gemini-3.5-flash \
  --api-key-file /path/to/gemini.key \
  --split dev \
  --target-lang zh \
  --deps-dir /path/to/simuleval_deps
```

Do not pass API keys through environment variables. Use a local key file and do
not commit it.

### 2026-07-04 EN->ZH Dev Results

All rows below use ACL6060 dev segmented source wavs, zh references, SimulEval
offline delays, `eval_latency_unit=char`, and `sacrebleu-tokenizer=zh`.

| run | rows | BLEU | LAAL | AL | AP | DAL | ATD | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| old Infinisst/RAG zh log | 468 | 51.316 | 6639.597 | 6639.597 | 1.034 | 6639.597 | 2215.145 | Taurus old log |
| OpenAI `gpt-audio-mini` | 468 | 51.438 | 6639.597 | 6639.597 | 1.668 | 6639.597 | 2277.545 | 20 refusal-like outputs |
| Gemini `gemini-3.5-flash` | 468 | 50.106 | 6639.597 | 6639.597 | 0.958 | 6639.597 | 2677.523 | 1 empty output |

Artifacts:

```text
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_dev_enzh_compare/
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_dev_enzh_openai_gpt_audio_mini/
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_dev_enzh_gemini_audio/
```

Tokenizer diagnostic on the same old `instances.log`:

| tokenizer | BLEU |
| --- | ---: |
| `zh` | 51.316 |
| `13a` | 3.479 |
| `intl` | 8.014 |
| `char` | 50.507 |

This explains why ACL6060 EN->ZH BLEU can look much higher than FLORAS scores:
the old ACL6060 number is text prediction vs text reference with Chinese
tokenization/character-like segmentation. It is not target-speech-ASR BLEU.

## 2026-07-24 ACL6060 SEGALE 重评（当前 canonical）

本节取代本文档中随后 `2026-07-23` 记录的 OmniSTEval、reference-based
XCOMET-XL 和 `weight_chars` 加权结论。旧记录只保留为历史实验说明，不能再用于
主表、看板或论文。

当前完整结果在：

```text
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_full_table.tsv
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_full_table.jsonl
```

它覆盖 En-Zh / En-De / En-Ja、`1x` / `1.25x` / `1.5x` 和 GPT Realtime /
Gemini Live / KIT Lecture Translator 共 27 个条件。全局 XCOMET-XL QE 值为
`0.7074453687`，来自全部 `10,765` 个 SEGALE 对齐单元的算术平均，包含空对齐的
零分。

### 对齐与质量协议

- 对齐使用 [Speech-to-Speech-Latency](https://github.com/SakaiXue6666/Speech-to-Speech-Latency)
  的 SEGALE 实现，固定 revision `d0041438abf097a1ec3055e7f09656ad6302f672`：
  spaCy 句切分、`sentence-transformers/LaBSE` contiguous span embedding、
  Vecalign 单调 many-to-many 对齐，`max_size=8`。Vecalign 的 adaptive skip
  penalty 允许显式 null alignment，因而不会把缺失 hypothesis 静默跳过。
- 上游 `src/evaluation/tgt_matching.py` 有一个 docstring 引号语法错误；本项目以
  `third_party_patches/speech-to-speech-latency/0001-fix-tgt-matching-docstring.patch`
  记录唯一的一行可复现修补，不改变算法。
- SEGALE 产出的 reference segment id 被保留为上游 LongYAAL matcher 所需的全局
  1-based id；每个 run 审计 source coverage，确保每个 ACL source segment 恰好
  对齐一次。
- BLEU 以 SEGALE 的 aligned `reference/hypothesis` 计算，保留原始标点；Zh=`zh`，
  Ja=`ja-mecab`，De=`13a`。空对齐也进入语料级 BLEU，不能通过过滤提高分数。
- XCOMET-XL 使用 QE / reference-free `source+hypothesis` 模式，**不输入 target
  reference**。主表用 COMET 官方算术平均，不做字符长度或任何其他加权。
  `source==""` 是 over-translation、`hypothesis==""` 是 under-translation；两类
  都被主动赋值 `0.0` 并纳入平均。全部 736 个 null alignment 已验证均为零分。
- LongYAAL 使用相同 SEGALE 对齐和 speed-scaled source audio (`offset` / `duration`
  除以 speedup)。空对齐没有定义 arrival latency，因此只从 LongYAAL 的 timing
  aggregate 略去，但保留在 BLEU/XCOMET 的质量惩罚中。所有 10,765 个 timing
  records 都是 `char_span_from_segale`、`skip_under_translation` 或
  `skip_over_translation`，没有 source-text fallback。

### 可复现入口与审计

```bash
scripts/run_acl6060_metric_pipeline.py \
  --artifact-base projects/acl6060_s2s_metrics_seed/artifacts \
  --speech-latency-repo /path/to/Speech-to-Speech-Latency \
  --run-segale --run-segale-longyaal --run-xcomet --reference-free-xcomet
```

关键可复用产物：

```text
artifacts/<run>/segale_alignment/ref.jsonl
artifacts/<run>/segale_alignment/hyp/aligned_spacy_hyp.jsonl
artifacts/<run>/segale_longyaal/{scores.resegmented.csv,summary.json}
artifacts/<run>/xcomet_xl/{input.jsonl,segments.jsonl,summary.json}
artifacts/acl6060_xcomet_xl_segale/{input_all.jsonl,scores_all.jsonl,summary_all.json}
```

最终审计：27/27 主表行均有 BLEU、XCOMET-XL、LongYAAL 和 Ending Offset；27 个
condition 唯一；combined XCOMET 输入/输出/SEGALE 对齐均为 10,765 行；source
coverage 无遗漏或重复；null score 非零数为 0。

逐候选 LaBSE matching trace (`*_aps_results.json`) 和逐 token LongYAAL trace
(`instances.resegmented.json`) 是可由上述 Git 输入确定性重建的运行中间件，保留在
本机/Aries 持久目录，不提交到 Git。

## 2026-07-23 ACL6060 3x3x3 Full Table Pipeline（历史）

目标表格覆盖 3 个 target language (`zh`, `de`, `ja`), 3 个 source speedup
(`1`, `1.25`, `1.5`), 3 个 system (OpenAI Realtime, Gemini Live, KIT Lecture
Translator)。统一输出路径:

```text
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_full_table.tsv
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_full_table.jsonl
```

Google Sheet:

https://docs.google.com/spreadsheets/d/1_HUxgwe8jqgoL9GRuVR1HZz__GOK2lfjFbqnFhweimM/edit?gid=0#gid=0

2026-07-23 已将 canonical 27 行写入 `ACL 60/60 Dev` sheet 的
`D2:H10`, `D12:H20`, `D22:H30`。写入后逐 cell 回读，数值与
`acl6060_full_table.tsv` 一致；9 个 KIT rows 均为
`ttsQualityMode=high_quality`。

一键 resume 脚本:

```bash
scripts/run_acl6060_full_table.sh
```

它按顺序执行:

- `scripts/run_acl6060_live_compare.sh`: 跑 OpenAI/Gemini full-wav live rows。
  当前 full-table 默认只跑 `chunk_ms=960` 和 `speed=1,1.25,1.5`；`--no-score`
  时也会把完整 5-row run copy 到 `artifacts/`，后续统一用 OmniSTEval 打分。
- `scripts/run_acl6060_kit_live_eval.py`: 跑 KIT rows。KIT 使用 target-language
  + English bilingual source-language 参数，例如 En-Zh 为
  `language=zh&language=en`, `mtLanguage=zh`, `audioLanguage=zh`,
  `format=mixed`, `ttsQualityMode=high_quality`。主 metric hypothesis 来自
  retrieved `tts:0` target speech 经过 `gpt-4o-mini-transcribe`，不使用 KIT
  web text。长 target wav 按真实 TTS chunk 边界分成最多 120 秒的 ASR
  windows，避免单请求 transcript output token cap 截断尾部。
- `scripts/run_acl6060_metric_pipeline.py`: 对所有已有 run 执行
  `scripts/run_acl6060_omnisteval.py`，生成 BLEU, `LongYAAL (CU)`,
  `ending_offset_ca_ms_mean`，并生成 XCOMET-XL segment 输入。
- `scripts/refresh_kit_auth.py`: 通过 KIT Dex email login 交互式刷新
  forward-auth cookie。密码只从 `getpass` 读取；输出文件原子写入并设为
  `0600`。
- `scripts/repair_acl6060_word_emissions.py`: 从保留的 raw API events 重建
  De whitespace-word emission timestamps，用于修复旧 run 中 transcript
  delta 在单词中间切分导致的 LongYAAL 错误。
- `scripts/repair_acl6060_kit_asr.py`: 复用已生成的 KIT target wav 和 TTS
  chunk metadata，把旧的 single-request long-audio ASR 替换为可 resume 的
  grouped-window ASR，不重跑 KIT streaming。

OmniSTEval/LongYAAL 细节:

- 使用 `omnisteval==0.1.10` longform evaluator。
- `speed_factor` rows 会把 `audio.yaml` 的 source offsets/durations 按
  `1/speed_factor` 缩放后再送入 LongYAAL。
- Zh/Ja 使用 char-level hypothesis units；De 使用 whitespace word units。
- De 的 emission time 使用完整单词最后一个字符所在 transcript delta 的
  source/wall timestamp。不能把任意 transcript delta 当作一个完整单词，也
  不能把 delta-level timestamps 直接截断到最终 word count。
- BLEU 保留 hypothesis/reference 原始标点；tokenizer 为 zh=`zh`,
  ja=`ja-mecab`, de=`13a`。
- 表中 `LongYAAL` 使用 `LongYAAL (CU)`；`Ending Offset` 使用
  `ending_offset_ca_ms_mean`。

OpenAI Realtime 约束:

- `GPT-realtime-translate` 使用专用
  `/v1/realtime/translations` translation session，不是 promptable voice
  agent。该接口不接受 `session.instructions`；表中 OpenAI config 因此没有
  custom prompt。
- runner 在发送任何 source audio 前等待 `session.updated` 确认
  `session.audio.output.language` 等于目标语言。update error 或 timeout
  会直接失败，避免无效语言输出进入表格。

Gemini Live session 约束:

- Gemini 会在 connection lifetime 结束前发送 `goAway(timeLeft=...)`。runner
  按最终 output transcript delta 的 idle time drain，不再被
  usage/keepalive events 延长到服务器强制 `1008` 关闭。
- 已完成的首批 segmented rows 在完整发送 480 秒首段、输出 drain 完成后
  收到 GoAway 关闭，再用新 connection 继续剩余 source；该关闭不代表翻译
  缺失。非 GoAway receiver exception 会计入 `error_count`。
- 官方 session lifecycle:
  https://ai.google.dev/gemini-api/docs/live-api/session-management

KIT target-speech ASR 约束:

- 12-14 分钟 target wav 单次调用 `gpt-4o-mini-transcribe` 会在 response
  output token cap 处半句截断。不能把该文本作为 full-wav hypothesis。
- 当前实现按 `audio_chunks.jsonl` 中的真实 TTS chunk 边界分组，每个窗口
  最多 120 秒，不硬切语音、不 overlap 重复；窗口文本顺序拼接并保留原标点。
- 每个 sample 持久化 `target_asr_windows.jsonl`，API 中断后可按
  `window_index` resume。

XCOMET-XL 细节:

- `scripts/build_acl6060_xcomet_input.py` 从 OmniSTEval resegmented segments
  生成 `src+hyp+ref` 输入；ACL dev 有人工 reference，所以这里用
  reference-based XCOMET-XL，不是之前 FLORAS 的 QE-lite diagnostic 分。
- `scripts/run_acl6060_xcomet_xl.py` 支持把 combined scores 按 `run_dir`
  拆回每个 artifact 的 `xcomet_xl/summary.json`。
- 27 个 rows 的 combined input、scores 和总览均已生成:

```text
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_xcomet_xl/input_all.jsonl
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_xcomet_xl/scores_all.jsonl
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_xcomet_xl/summary_all.json
```

当前状态:

- 27/27 行均已有 BLEU、XCOMET-XL、LongYAAL 和 Ending Offset。最终 TSV
  和 JSONL 是本表的 canonical results。
- 27 行共包含 135 个 full-wav system outputs。自动审计确认每行 5 个
  sample、全部 OpenAI/Gemini live API error 为 0、emission timestamp
  单调、source speed 缩放一致、目标语言正确，并保留 hypothesis/reference
  标点。
- `Unbabel/XCOMET-XL` 对 27 x 468 = 12636 个 `src+hyp+ref` segment 完成
  reference-based scoring。按 `weight_chars` 加权的 combined mean 为
  `0.7877888664`；每行 `xcomet_xl/summary.json` 都由 468 个 segment
  重新聚合并与最终表核对。
- En-De OpenAI 三行的错误负 LongYAAL 已修复。修复后的 `LongYAAL (CU)`
  为 speed `1`=`4605.7919`, `1.25`=`4990.0444`,
  `1.5`=`9788.2364`；BLEU 不变。
- KIT 的 `2022.acl-long.367` 在 speed `1.25` 三种目标语言中都只产生了
  53-94 个 timing units。这不是 target ASR 截断：输入 audio POST 均成功，
  已取回全部 TTS chunks，窗口 ASR 也完整；KIT source ASR 长时间不稳定，
  导致 MT/TTS 只消费少量已稳定文本。这是实际 S2S failure，结果中保留。
- XCOMET-XL 使用 model revision
  `6a123c5e8e6dccab25e5fcffa3c8b417abadb462`，在 hyper01 的 3 张 H200
  上分片运行。环境为 `unbabel-comet==2.2.7`,
  `torch 2.11.0+cu130`, `torchvision 0.26.0+cu130`,
  `transformers 4.40.2`, `huggingface-hub 0.23.5`。临时 HF token、任务
  container 和远端中间目录均已删除。

## 2026-07-23 KIT Low-Latency One-Cell A/B

对正式表中的 `En-Zh / 1x / KIT` 单格做了控制变量实验。两组均使用
`chunk=960ms`, `format=mixed`, `language=zh,en`,
`smartChaptering=online_dynamic`, private availability，以及同一套 target
speech `gpt-4o-mini-transcribe` windowed ASR。唯一变量是
`ttsQualityMode=high_quality` 或 `low_latency`。comparison builder 会比较
两个 tracked `run_config.json` 并在差异字段不等于
`kit_tts_quality_mode` 时直接失败。high-quality baseline 是正式表已有的:

```text
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_live_enzh_kit_chunk960_speed1/
```

builder 还会把该 artifact 的 BLEU `35.4007`, XCOMET-XL `0.680956`,
LongYAAL `85358.7596` 和 Ending Offset `166172.1296` 与
`acl6060_full_table.jsonl` 中唯一的 `En-Zh / 1x / KIT` canonical row
逐项核对，不一致时直接失败。

| metric | high_quality | low_latency | low - high |
| --- | ---: | ---: | ---: |
| BLEU | 35.4007 | 35.5727 | +0.1720 |
| XCOMET-XL | 0.680956 | 0.678843 | -0.002113 |
| LongYAAL (CU, ms) | 85358.7596 | 74863.3737 | -10495.3859 |
| Ending Offset (CA, ms) | 166172.1296 | 156544.5680 | -9627.5616 |
| mean TTS audio chunks | 70.0 | 90.8 | +20.8 |
| mean target audio duration (ms) | 558980.0 | 589235.0 | +30255.0 |
| mean prediction units | 2269.2 | 2256.4 | -12.8 |

结论:

- BLEU 上升 `0.1720`，XCOMET-XL 下降 `0.0021`，方向不一致。这里只有 5
  个彼此相关的 talk，未做 paired significance test，因此只能说没有看到
  明确的质量变化，不能断言两者统计等价或 low latency 无质量损失。
- aggregate LongYAAL 观察值降低约 `10.50s`，Ending Offset 降低约
  `9.63s`，但不能直接解释为纯粹的 TTS mode latency gain。
- `2022.acl-long.367` 在两种 mode 下都是严重 partial-output failure。
  low latency 只有 10 个目标音频块和 160 个字符，high quality 也只有 9
  块和 145 个字符；两者都遗漏了大部分 11.6 分钟输入。该失败按正式协议
  保留在五条 aggregate 中。
- 逐 talk Ending Offset 可与五条 aggregate 的算术均值对齐。除去无法
  解释 latency 的失败样本 `367`，其余四条中两条改善、两条变差:

| talk | high_quality | low_latency | Ending Offset delta (ms) |
| --- | ---: | ---: | ---: |
| `2022.acl-long.268` | 167352.2440 | 130791.2540 | -36560.9900 |
| `2022.acl-long.367` | 173025.4850 | 165339.7030 | -7685.7820 (failure; non-informative) |
| `2022.acl-long.590` | 136020.3600 | 149350.8840 | +13330.5240 |
| `2022.acl-long.110` | 191632.6510 | 167793.6190 | -23839.0320 |
| `2022.acl-long.117` | 162829.9080 | 169447.3800 | +6617.4720 |

- 当前 target-unit timing proxy 假设 ASR 字符在 target audio 上均匀分布，
  再把每个字符吸附到覆盖它的 TTS chunk arrival。low latency 平均 chunk
  数为 `90.8`，high quality 为 `70.0`；时间量化粒度本身与 treatment
  变量一起改变，所以 LongYAAL 存在 chunk-granularity confound。
- `delays` 还会被 clamp 到 `source_length`，因此 source 结束后才到达的
  target tail 在两种 mode 下都会被低估。per-talk TSV 中的 LongYAAL 是把
  talk 隔离后重新运行 OmniSTEval 的 non-additive diagnostic，不能视为
  joint 468-segment aggregate LongYAAL 的贡献分解；`367` 的隔离
  LongYAAL 尤其没有解释意义。
- 5/5 samples 均确认 `pauseStatus.ok`, `TTS-finish`，且 KIT 已提供的所有
  target audio chunks 均被抓取并用 grouped-window ASR 转写。这只排除了
  fetch/ASR 额外截断，不代表 KIT 对 source 内容的覆盖完整，`367` 明确
  不是完整输出。
- 正式 27 行 canonical table 暂时仍保留 `high_quality`。本实验只覆盖一个
  En-Zh/1x cell，尚不足以把其他 KIT 行统一切换为 low latency。

Tracked comparison:

```text
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_kit_enzh_speed1_quality_mode_comparison.tsv
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_kit_enzh_speed1_quality_mode_comparison.json
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_kit_enzh_speed1_quality_mode_comparison_per_talk.tsv
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_live_enzh_kit_low_latency_chunk960_speed1/
```

使用 tracked run artifacts、ACL6060 HF/RASST dataset 和
`omnisteval==0.1.10` runtime 可复现 comparison；不是只靠 Git checkout
即可运行:

```bash
python scripts/build_acl6060_kit_quality_mode_comparison.py \
  --dataset-root /tmp/rasst_main_result_data \
  --omnisteval-bin /path/to/omnisteval
```

Targeted verification:

```text
tests/test_acl6060_kit_eval.py + tests/test_acl6060_stream_eval.py: 11 passed
comparison builder: 5 instances, 5 responses, 468 XCOMET-XL segments
config diff: kit_tts_quality_mode only
canonical baseline metrics: exact at stored table precision
Ending Offset: per-talk mean reconciles to aggregate in both modes
```

完整目标语音和逐 chunk 原始记录已移到持久化本机 staging:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/acl6060_kit_live_sweep/enzh_kit_chunk960_speed1_low_latency
```

该 201 MB raw audio bundle 尚未上传 Hugging Face，状态为
`PENDING_HF_UPLOAD`。Git artifact 保留 5-row instances/responses、配置、
OmniSTEval 和 468 条 XCOMET-XL 输入/分数。

## 2026-07-24 ACL6060 En-Zh Audio Detail

为展示 `1.25x` 和 `1.5x` 的定性细节，选择两条稳定 talk
`2022.acl-long.268`、`2022.acl-long.590`，生成同页横向对比
GPT/Gemini/KIT 的本地 detail 页面。已知 KIT partial-output failure
`2022.acl-long.367` 不作为展示样例。

页面入口:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/acl6060_enzh_audio_samples_all_systems/index.html
```

数据语义:

- 每个 speed/talk 有一个 60 秒 sped English source，以及 GPT、Gemini、KIT
  三个中文 target audio player，共 4 组、16 个 player。
- GPT/Gemini 历史 canonical full-wav raw logs 已将 base64 audio payload
  脱敏，无法从旧日志恢复目标语音。因此重新使用相同 model、`target=zh`,
  `chunk=960ms`, speed 配置，对相同 first-60s sped source 做定性重跑。
  8 次 API 请求 `error_count` 均为 0；这些音频不是 canonical full-wav
  session 的原始音频，不能用于替换正式表分数。
- KIT 使用 canonical full-wav run 中已经持久化的
  `target_tts.wav` 前 60 秒；配置仍为 `format=mixed`,
  `language=zh,en`, `ttsQualityMode=high_quality`。这是 full-talk target
  timeline 上的时间裁切，不与 GPT/Gemini 的 60 秒 source window 对齐，
  可能覆盖更少 source 内容；页面只用于 speech-quality inspection，不是
  source-aligned ranking。KIT text panel 也明确标记为 full-talk
  reference-aligned excerpt，并非该 60 秒 audio 的精确 transcript。
- 页面显示 canonical full-wav BLEU/XCOMET 仅作系统背景；它们不是用
  60 秒展示样例重新计算的指标。
- 展示 MP3 统一为 24kHz mono、64kbps，并做
  `loudnorm=I=-20:TP=-2:LRA=11`，便于公平试听；raw WAV 保持不变。
- 每个 detail 同时展示 ACL English source transcript、人工中文 reference
  和系统 target transcript。页面和 manifest 共约 8.5 MB；GPT/Gemini raw
  rerun 约 48 MB。

可复现 builder:

```bash
python scripts/build_acl6060_enzh_audio_details.py \
  --rerun-root /Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/acl6060_enzh_audio_sample_reruns \
  --output-dir /Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/acl6060_enzh_audio_samples_all_systems
```

生成音频属于 reusable data artifact，当前仍在本机 persistent staging:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/acl6060_enzh_audio_sample_reruns
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/acl6060_enzh_audio_samples_all_systems
```

预定 Hugging Face dataset repo 为
`gavinlaw/acl6060-s2s-audio-samples-en-zh`，当前状态
`PENDING_HF_UPLOAD`；本次没有执行 HF 创建或上传。

## Related KIT Lecture Translator Work

KIT Lecture Translator was first explored under the FLORAS live benchmark. For
the ACL6060 full table above, use `scripts/run_acl6060_kit_live_eval.py` and
score only retrieved target speech through ASR. Do not compare a KIT web-event
text capture directly with ACL6060 rows.

Current KIT status on 2026-07-06:

- The tracked KIT/GPT/Gemini/Seed dashboard is a 60s smoke/debug artifact for
  text extraction and Chinese BLEU tokenization.
- A full-source KIT attempt on the FLORAS EN->ZH sample used a default
  low-latency online configuration and was interrupted after 330.0s of the
  1072.63s source, then paused. It is not a formal score.
- KIT `format=mixed` rows are valid S2S smoke rows when the hypothesis is
  retrieved target speech scored through ASR. Do not use KIT web-event text as
  the main hypothesis, because displayed text may be revised.
- The current 60s target-speech-ASR KIT smoke best row is
  `mixed_high_quality_no_post`: BLEU 25.94, chrF 22.72, CER 0.717 on the
  selected FLORAS EN->ZH clip. The explicit minimal
  `language=en, mtLanguage=zh, audioLanguage=zh, format=online,
  ttsQualityMode=high_quality` run scored lower: BLEU 20.47, chrF 19.83,
  CER 0.767.
- A source-speech speed=1.5 smoke with `format=mixed`,
  `ttsQualityMode=high_quality`, and 1.92s chunks scored BLEU 23.26, chrF 21.49,
  CER 0.717; target speech was 69.58s for a 40.03s source stream. The companion
  FLORAS speed=1.5 dashboard compares this KIT exact-60s smoke row against
  GPT/Gemini/Seed full-run generated target wavs cropped to their first 60s and
  re-transcribed, so treat that table as a target-audio crop view rather than a
  source-aligned formal ranking.
- Before a formal KIT comparison, inspect or sweep the KIT product settings and
  decide whether KIT can be scored from target speech ASR. If only web-event TTS
  text is available, label it as text-only.

See:

```text
projects/floras_live_s2s_benchmark/README.md
projects/floras_live_s2s_benchmark/RESULTS.md
docs/handoff_s2s_omni.md
```
