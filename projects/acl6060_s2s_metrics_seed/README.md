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
