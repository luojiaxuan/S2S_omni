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

## ACL6060 GPT/Gemini Offline Reproduction

`run_acl6060_llm_audio_eval.py` runs an offline speech-to-text translation
baseline over the ACL6060 segmented source wavs. It writes SimulEval-compatible
`instances.log` records and scores them with the same SimulEval + sacreBLEU
path used by the old ACL6060 logs.

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
