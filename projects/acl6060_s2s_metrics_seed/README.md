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

## 2026-07-23 ACL6060 3x3x3 Full Table Pipeline

目标表格覆盖 3 个 target language (`zh`, `de`, `ja`), 3 个 source speedup
(`1`, `1.25`, `1.5`), 3 个 system (OpenAI Realtime, Gemini Live, KIT Lecture
Translator)。统一输出路径:

```text
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_full_table.tsv
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_full_table.jsonl
```

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
