# Hibiki-Zero Backlog-Aware S2S Route

This route treats Qwen3-Omni end-to-end talker issues as non-blocking. The
working baseline is Hibiki-Zero for `fr/es/pt/de -> en`, with a cascade teacher
that creates high-quality S2S supervision:

```text
source speech chunks
  -> Qwen3-Omni thinker-only compressed English chunk text
  -> MOSS/Higgs/Qwen3-TTS English target speech
  -> English MFA
  -> per-chunk speech-to-speech SFT manifest
```

## Schema

Sample-level manifests use:

```text
sample_id
src_lang
source_audio_chunks
source_text
reference_en_text
compressed_en_text
target_en_wav
target_chunk_wavs
mfa_boundaries
duration_budget_s
speech_s2s_rtf
quality_gates
```

Each `source_audio_chunks` item contains:

```text
chunk_index
source_audio
source_duration_s
source_text
reference_en_text
compressed_en_text
duration_budget_s
```

## Local Script Chain

Normalize source data:

```bash
python scripts/hibiki_zero_prepare_sources.py \
  --input /path/to/x_to_en_source.jsonl \
  --output-dir /data/runs/hz/data \
  --max-per-lang 100 \
  --split
```

If the source is a TSV with `audio/src_text/tgt_text/src_lang/tgt_lang` columns,
convert it first:

```bash
python scripts/hibiki_zero_tsv_to_source_manifest.py \
  --input /path/to/covost_or_st.tsv \
  --audio-root /path/to/dataset/root \
  --output /data/runs/hz/raw_source.jsonl \
  --max-per-lang 100
```

For smoke tests without CoVoST access, export parallel FLEURS source speech and
English reference text:

```bash
python scripts/hibiki_zero_fleurs_to_source_manifest.py \
  --output /data/runs/hz/fleurs_source.jsonl \
  --audio-dir /data/runs/hz/fleurs_audio \
  --split validation \
  --max-per-lang 5
```

Expand source rows with backlog speed stress before teacher generation:

```bash
python scripts/hibiki_zero_speed_stress_sources.py \
  --input /data/runs/hz/data/source_manifest.jsonl \
  --output /data/runs/hz/data/source_manifest_speed.jsonl \
  --audio-dir /data/runs/hz/data/source_speed_wav \
  --speed-factors 1.0,1.35,1.7,2.0 \
  --speed-assignment cycle
```

Generate compressed English teacher text:

```bash
python scripts/hibiki_zero_generate_teacher_text.py \
  --input /data/runs/hz/data/source_manifest_speed.jsonl \
  --output /data/runs/hz/teacher/teacher_manifest.jsonl \
  --backend transformers_omni \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct
```

For an OpenAI-compatible Qwen3-Omni or text fallback endpoint:

```bash
python scripts/hibiki_zero_generate_teacher_text.py \
  --input /data/runs/hz/data/source_manifest_speed.jsonl \
  --output /data/runs/hz/teacher/teacher_manifest.jsonl \
  --backend openai \
  --base-url http://127.0.0.1:30000/v1 \
  --api-key EMPTY \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct
```

Generate English target speech through a TTS HTTP endpoint:

```bash
python scripts/hibiki_zero_generate_tts_targets.py \
  --input /data/runs/hz/teacher/teacher_manifest.jsonl \
  --output-dir /data/runs/hz/tts \
  --urls http://127.0.0.1:18112/v1/audio/speech,http://127.0.0.1:18113/v1/audio/speech \
  --backend moss_tts_http \
  --workers 4
```

Run English MFA on the generated corpus:

```bash
mfa align \
  /data/runs/hz/tts/mfa_corpus \
  english_us_arpa \
  english_us_arpa \
  /data/runs/hz/mfa_aligned \
  --clean \
  --overwrite
```

Slice into streaming target chunks and apply source speed stress:

```bash
python scripts/hibiki_zero_slice_mfa_chunks.py \
  --tts-manifest /data/runs/hz/tts/tts_manifest.jsonl \
  --textgrid-dir /data/runs/hz/mfa_aligned \
  --output-dir /data/runs/hz/sft \
  --speed-factors 1.0,1.35,1.7,2.0 \
  --speed-assignment cycle
```

Validate and render samples:

```bash
python scripts/hibiki_zero_validate_manifest.py \
  --manifest /data/runs/hz/sft/sft_sample_manifest.jsonl \
  --output /data/runs/hz/sft/validation.json

python scripts/hibiki_zero_render_html.py \
  --manifest /data/runs/hz/sft/sft_sample_manifest.jsonl \
  --output /data/runs/hz/index.html
```

Run Hibiki-Zero zero-shot baseline:

```bash
python scripts/hibiki_zero_run_baseline.py \
  --manifest /data/runs/hz/sft/sft_sample_manifest.jsonl \
  --output-dir /data/runs/hz/baseline \
  --max-records 20
```

## Aries Smoke

Set up the reusable container:

```bash
ssh -T -o RemoteCommand=none -o RequestTTY=no aries \
  'cd /mnt/data3/jiaxuanluo/S2S_omni && CONTAINER_NAME=sglang-omni-jiaxuan-aries-hibiki-zero bash scripts/setup_hibiki_zero_aries.sh'
```

Run a 20-row smoke after setting a real `SOURCE_MANIFEST` for
`fr/es/pt/de -> en`:

```bash
ssh -T -o RemoteCommand=none -o RequestTTY=no aries \
  'cd /mnt/data3/jiaxuanluo/S2S_omni && \
   CONTAINER_NAME=sglang-omni-jiaxuan-aries-hibiki-zero \
   SOURCE_MANIFEST=/path/to/fr_es_pt_de_to_en.jsonl \
   TTS_URLS=http://127.0.0.1:18112/v1/audio/speech \
   MFA_BIN=/path/to/mfa \
   bash scripts/run_hibiki_zero_smoke_aries.sh'
```

## Current Training Boundary

The repo now produces sample-level and turn-level S2S manifests. The actual
Hibiki-Zero SFT adapter is intentionally a separate step because the public
Hibiki-Zero repo is inference-oriented. The next implementation step is to
inspect the installed package and wire `sft_turn_manifest.jsonl` into a
`moshi-finetune` LoRA job over Mimi/source/target streams.

The acceptance order is:

1. zero-shot baseline report on the same dev/test manifest
2. teacher-data smoke accepted rate at least `70%`
3. 12.5k dataset with less than `1%` missing or corrupt wavs
4. SFT candidate reduces `speech_s2s_rtf` violation rate without more than `5%`
   relative ASR-BLEU/chrF drop
