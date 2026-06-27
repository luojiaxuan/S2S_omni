# S2S_omni

Infra for adaptive semantic compression in streaming speech-to-speech translation.

The project is intentionally separated from `/home/sglang-omni/sglang-omni`.
The first milestone is text-side and transcript-side infrastructure:

- build stress manifests for fast-source streaming conditions
- generate compression-aware SFT JSONL
- run heuristic and LLM-as-judge evaluation
- keep Qwen3-Omni speech serving and LoRA SFT entrypoints ready for the next pass

## Problem

When the source speaker is too dense or too fast, a speech-to-speech translator has
three bad options: accumulate lag, speed up the target audio until it is hard to
listen to, or silently drop content. This repo studies a fourth option:
meaning-preserving compression under listenability constraints.

The desired behavior is not generic summarization. The system must preserve core
propositions, entities, numbers, terms, speaker intent, and discourse-critical
relations while compressing repeated, hedged, low-information, or non-critical
content.

## Layout

```text
configs/
  compression_policy.yaml       # default budget and listenability policy
  qwen3_omni_lora_sft.yaml      # text-side LoRA SFT config template
data/examples/
  seed_segments.jsonl           # tiny smoke-test corpus
docs/
  eval_plan.md                  # literature-backed metric plan
  hibiki_zero_backlog_route.md  # Hibiki-Zero cascade data and baseline route
  remote_artifacts.md           # remote data/checkpoint path index
s2s_omni/
  schema.py                     # dataclasses for samples, timing, budgets
  io.py                         # JSONL/YAML helpers
  prompts.py                    # teacher and judge prompts
  metrics.py                    # heuristic compression/listenability metrics
  judge.py                      # LLM-as-judge orchestration
  llm_client.py                 # minimal OpenAI-compatible chat client
  sft.py                        # chat-format and JSONL SFT export
  streaming.py                  # speed-up and lag-budget manifest transforms
  hibiki_zero.py                # Hibiki-Zero X->English S2S manifest helpers
scripts/
  make_stress_manifest.py       # expand samples into speed/budget conditions
  build_gigaspeech_sft.py       # build GigaSpeech warm-up and teacher-request data
  generate_teacher_labels.py    # local/OpenAI-compatible teacher label generation
  merge_teacher_labels.py       # merge teacher shards into accepted SFT JSONL
  make_sft_jsonl.py             # create SFT training JSONL
  evaluate_predictions.py       # score model outputs
  run_qwen3_omni_baseline.py    # hit an OpenAI-compatible Qwen3-Omni server
  run_teacher_generation_shard.sh # resumable single-GPU teacher shard runner
  summarize_teacher_labels.py   # accepted-rate and compression-ratio summaries
  train_text_lora_sft.py        # dependency-checked LoRA SFT entrypoint
  watch_and_merge_teacher.sh     # wait for shards, merge labels, summarize
  verify_split_integrity.py      # verify held-out base_id split isolation
  hibiki_zero_prepare_sources.py # normalize fr/es/pt/de -> en source manifests
  hibiki_zero_fleurs_to_source_manifest.py # export FLEURS parallel smoke data
  hibiki_zero_tsv_to_source_manifest.py # convert ST TSV rows to source schema
  hibiki_zero_speed_stress_sources.py # expand source manifests with speed budgets
  hibiki_zero_generate_teacher_text.py # Qwen3-Omni compressed English teacher
  hibiki_zero_generate_tts_targets.py # MOSS/Higgs/Qwen3-TTS target speech HTTP
  hibiki_zero_slice_mfa_chunks.py # English MFA word-boundary target chunking
  hibiki_zero_run_baseline.py    # run official hibiki-zero generate baseline
  run_smoke.sh                  # quick local/remote smoke test
```

## Quick Start

From this directory:

```bash
python3 scripts/make_stress_manifest.py \
  --input data/examples/seed_segments.jsonl \
  --output work/stress_manifest.jsonl \
  --speed-factors 1.0,1.35,1.7

python3 scripts/make_sft_jsonl.py \
  --input work/stress_manifest.jsonl \
  --output work/sft.jsonl

python3 scripts/evaluate_predictions.py \
  --gold work/stress_manifest.jsonl \
  --pred data/examples/seed_segments.jsonl \
  --output work/eval.jsonl \
  --summary work/eval_summary.json
```

Or run:

```bash
bash scripts/run_smoke.sh
```

## LLM Judge

The judge uses an OpenAI-compatible chat completion endpoint through standard
library HTTP. Set:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
export S2S_JUDGE_MODEL=your-judge-model
```

Then pass `--judge` to `scripts/evaluate_predictions.py`.

## Evaluation

Before SFT/RL, run cheap metrics on teacher labels and baselines. The current
suite reports reference-length ratio, hard-budget ratio, budget violations,
number/term recall, rough lexical F1, estimated speech-rate pressure, end lag,
and p50/p90/p95/max summaries. If `sacrebleu` is installed it also adds corpus
BLEU and chrF as classic reference metrics.

Teacher labels can be evaluated directly:

```bash
python scripts/evaluate_predictions.py \
  --gold work/gigaspeech_pilot/compression_teacher_requests.jsonl \
  --pred work/gigaspeech_pilot/compression_teacher_labels.jsonl \
  --output work/gigaspeech_pilot/compression_teacher_eval.jsonl \
  --summary work/gigaspeech_pilot/compression_teacher_eval_summary.json
```

Add `--judge` for LLM-as-judge once an OpenAI-compatible judge endpoint is up.
The judge rubric is compression-aware: it allows paraphrase and safe omission,
but penalizes critical omissions, hallucinations, number/entity/negation errors,
and outputs that would be hard to listen to in real time.

See `docs/eval_plan.md` for the paper-backed metric plan and the SFT gate.

## Qwen3-Omni Baseline

Start the Qwen3-Omni speech or text server from the `sglang-omni` serving stack,
then call:

```bash
python3 scripts/run_qwen3_omni_baseline.py \
  --base-url http://127.0.0.1:30000/v1 \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --input work/stress_manifest.jsonl \
  --output work/qwen3_baseline.jsonl \
  --text-only
```

For audio output, omit `--text-only` and make sure the server is in speech mode.

vLLM is intentionally not part of this repo's inference path.

## Hibiki-Zero Backlog-Aware Baseline

The current non-Omni-codec route uses Hibiki-Zero as the S2S baseline for
`fr/es/pt/de -> en`. It builds cascade teacher data:

```text
source speech -> Qwen3-Omni thinker-only compressed English text
              -> MOSS/Higgs/Qwen3-TTS English speech
              -> English MFA chunk slicing
              -> Hibiki-style S2S SFT manifests
```

Primary scripts:

```bash
python scripts/hibiki_zero_prepare_sources.py --help
python scripts/hibiki_zero_generate_teacher_text.py --help
python scripts/hibiki_zero_generate_tts_targets.py --help
python scripts/hibiki_zero_slice_mfa_chunks.py --help
python scripts/hibiki_zero_run_baseline.py --help
```

On Aries, use:

```bash
CONTAINER_NAME=sglang-omni-jiaxuan-aries-hibiki-zero \
  bash scripts/setup_hibiki_zero_aries.sh
```

Then run the smoke chain with a real `SOURCE_MANIFEST`:

```bash
CONTAINER_NAME=sglang-omni-jiaxuan-aries-hibiki-zero \
SOURCE_MANIFEST=/path/to/fr_es_pt_de_to_en.jsonl \
TTS_URLS=http://127.0.0.1:18112/v1/audio/speech \
  bash scripts/run_hibiki_zero_smoke_aries.sh
```

See `docs/hibiki_zero_backlog_route.md` for the full data/eval plan. The
Hibiki-Zero SFT adapter itself is the next step after verifying baseline and
teacher-data smoke, because the public Hibiki-Zero package is primarily
inference-oriented.

## RASST Soft-Wav E2E SFT

This is the current end-to-end speech-output route. It uses plain RASST zh
baseline data, not the term-map/tagged RASST variants:

```text
train: /mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline.jsonl
dev:   /mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline_dev.jsonl
```

The teacher target audio is generated with the original
`Qwen/Qwen3-Omni-30B-A3B-Instruct` audio-output path. It does not use Qwen3-TTS.
For each RASST row, the non-empty assistant chunks are concatenated, spoken by
Qwen3-Omni, and the generated Omni-compatible codec codes are captured before
`code2wav`. MFA aligns the generated full target wav to the concatenated
Mandarin target text; the aligned time spans are then cut back into per-chunk
target wav/code labels.

On Aries, from a fresh clone of this repo:

```bash
USE_USER_SITE=1 \
USER_BASE=/mnt/data/jiaxuanluo/python_userbase/s2s_omni_softwav \
PIP_CACHE_DIR=/mnt/data/jiaxuanluo/.cache/pip \
TMPDIR=/mnt/data/jiaxuanluo/tmp \
  bash scripts/setup_aries_softwav_env.sh

bash scripts/setup_aries_mfa_env.sh

export PYTHONUSERBASE=/mnt/data/jiaxuanluo/python_userbase/s2s_omni_softwav
export PATH=/mnt/data/jiaxuanluo/python_userbase/s2s_omni_softwav/bin:$PATH
export HF_HOME=/mnt/data3/jiaxuanluo/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export XDG_CACHE_HOME=/mnt/data/jiaxuanluo/.cache
export TMPDIR=/mnt/data/jiaxuanluo/tmp
export MFA_BIN=$(/mnt/data/jiaxuanluo/micromamba/bin/micromamba run -n mfa \
  python -c 'import shutil; print(shutil.which("mfa"))')

bash scripts/run_rasst_softwav_smoke_aries.sh
```

The smoke run does:

1. Generate 20 full-target Qwen3-Omni wav/code labels.
2. Run MFA Mandarin alignment.
3. Build a turn-level soft-wav manifest with real source speed-up.
4. Run a processor/tokenization smoke.
5. Run one optimizer step of thinker+talker LoRA SFT.

Full run:

```bash
nohup bash scripts/run_rasst_softwav_full_aries.sh \
  > /mnt/data/jiaxuanluo/S2S_omni_runs/rasst_softwav_full.log 2>&1 &
```

The trainer initializes from the original Qwen3-Omni, freezes base weights and
`code2wav`, and trains thinker+talker LoRA with text CE, codec CE, soft-code2wav
wav/STFT loss, and an EOS loss for empty target chunks.

## Omni-Compatible wav2codec

This path trains an inverter for the Qwen3-Omni self-domain codec:

```text
Omni-generated target wav -> Omni talker codes -> frozen Omni code2wav
```

The pair generator freezes `Qwen/Qwen3-Omni-30B-A3B-Instruct`, runs normal
speech-to-speech translation on GigaSpeech English source audio, and monkeypatches
`code2wav.chunked_decode` to capture the exact `talker_codes` used for the
returned wav.

Build split-isolated candidate manifests:

```bash
python scripts/build_omni_codec_pair_manifest.py \
  --tsv /mnt/taurus/data/siqiouyang/datasets/gigaspeech/train_xl_case_ft-qwen2.5-32b-instruct_marked_mfa_punc_asr.tsv \
  --output-dir /mnt/data2/jiaxuanluo/S2S_omni/work/omni_s2s_codec_pairs_25k_$(date +%Y%m%d)/manifests \
  --target-counts train=25000,dev=500,test=500
```

Generate wav/code pairs on aries or taurus:

```bash
MODE=generate SPLIT=train NUM_SHARDS=8 SHARD_INDEX=0 \
  bash scripts/run_wav2codec_pipeline_slurm.sh
```

Audit 8 generated pairs by decoding captured gold codes back through frozen
Omni code2wav:

```bash
MODE=audit SPLIT=train bash scripts/run_wav2codec_pipeline_slurm.sh
```

Training gates:

```bash
MODE=train_overfit bash scripts/run_wav2codec_pipeline_slurm.sh
MODE=train_smoke bash scripts/run_wav2codec_pipeline_slurm.sh
MODE=train_full bash scripts/run_wav2codec_pipeline_slurm.sh
```

Evaluate a checkpoint and render a listening page:

```bash
MODE=eval SPLIT=test bash scripts/run_wav2codec_pipeline_slurm.sh
```

The default remote paths are:

```text
/mnt/data2/jiaxuanluo/S2S_omni/work/omni_s2s_codec_pairs_25k_YYYYMMDD
/mnt/data2/jiaxuanluo/S2S_omni/checkpoints/wav2omni_codec_25k_YYYYMMDD
```

## GigaSpeech Data

The first data source is the existing GigaSpeech S2TT table on taurus:

```text
/mnt/taurus/data/siqiouyang/datasets/gigaspeech/train_xl_case_ft-qwen2.5-32b-instruct_marked_mfa_punc_asr.tsv
```

The builder first splits by original GigaSpeech `base_id`, then emits three
kinds of records inside each split:

- `faithful_sft.jsonl`: phase-0 warm-up, source transcript to full translation
- `pass_through_sft.jsonl`: speed-stressed chunks whose faithful translation
  still fits the default speech real-time budget, so the target stays unchanged
- `compression_teacher_requests.jsonl`: speed-stressed requests for a teacher
  model to create compressed translations

Training prompts never include the reference translation. Teacher requests do
include it, because the teacher is asked to compress a faithful translation under
a target duration/ratio budget.

Pilot split command:

```bash
python3 scripts/build_gigaspeech_sft.py \
  --max-rows 100000 \
  --faithful-limit 4000 \
  --pass-through-limit 2000 \
  --teacher-limit 2000 \
  --dev-pass-through-limit 300 \
  --test-pass-through-limit 300 \
  --dev-teacher-limit 300 \
  --test-teacher-limit 300 \
  --split-ratios train=0.9,dev=0.05,test=0.05 \
  --out-dir work/gigaspeech_pilot_split
```

The split pilot scanned 100,000 rows and found 98,990 usable rows. It wrote
train/dev/test files under `splits/`. The split unit is `base_id`, so different
speed variants of the same utterance cannot cross train/dev/test.

Compression is decided with a speech real-time factor, not source speed alone:

```text
s2s_rtf = faithful_target_speech_duration_at_default_rate / source_chunk_wall_duration
```

If `s2s_rtf <= 1.0`, the record goes to `pass_through_sft.jsonl` and the full
translation is the supervised target. If `s2s_rtf > 1.0`, the record goes to
`compression_teacher_requests.jsonl` with a hard maximum target length derived
from the default speech rate and source chunk duration. This teaches the model
to compact only when the target speech would otherwise miss the next source
chunk.

Verify split isolation before reporting held-out numbers:

```bash
python scripts/verify_split_integrity.py \
  --split train=work/gigaspeech_pilot_split/splits/train/compression_teacher_requests.jsonl \
  --split dev=work/gigaspeech_pilot_split/splits/dev/compression_teacher_requests.jsonl \
  --split test=work/gigaspeech_pilot_split/splits/test/compression_teacher_requests.jsonl \
  --output work/gigaspeech_pilot_split/split_integrity_teacher_requests.json
```

Root-level files under `work/gigaspeech_pilot_split/` are train aliases kept for
older commands. Use `splits/dev` and `splits/test` for held-out evaluation.

## Teacher Label Generation

Teacher labels use the local Qwen3-Omni thinker by default. The teacher prompt
can see the faithful reference translation; the emitted SFT records rebuild the
user prompt without the reference so the student cannot leak from the teacher
context.

Quick smoke:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=6 python scripts/generate_teacher_labels.py \
  --input work/gigaspeech_pilot/compression_teacher_requests.jsonl \
  --output work/gigaspeech_pilot/compression_teacher_labels_smoke.jsonl \
  --sft-output work/gigaspeech_pilot/compression_sft_smoke.jsonl \
  --backend transformers \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --max-records 20 \
  --max-new-tokens 160 \
  --log-every 5
```

Two-GPU resumable generation on b200:

```bash
mkdir -p logs
nohup env GPU=6 NUM_SHARDS=2 SHARD_INDEX=0 \
  bash scripts/run_teacher_generation_shard.sh \
  > logs/teacher_generation_shard0.log 2>&1 &

nohup env GPU=7 NUM_SHARDS=2 SHARD_INDEX=1 \
  bash scripts/run_teacher_generation_shard.sh \
  > logs/teacher_generation_shard1.log 2>&1 &
```

Check progress:

```bash
ps -u "$(whoami)" -o pid,cmd | grep "[p]ython scripts/generate_teacher_labels.py"
for i in 0 1; do
  wc -l work/gigaspeech_pilot/compression_teacher_labels_shard${i}.jsonl
  grep "generated" logs/teacher_generation_shard${i}.log | tail -n 3
done
```

Summarize current label quality:

```bash
python scripts/summarize_teacher_labels.py --pretty \
  work/gigaspeech_pilot/compression_teacher_labels_shard0.jsonl \
  work/gigaspeech_pilot/compression_teacher_labels_shard1.jsonl
```

Optional watcher that merges automatically once all shard processes exit:

```bash
nohup bash scripts/watch_and_merge_teacher.sh \
  > logs/teacher_generation_merge.log 2>&1 &
```

Merge accepted labels after both shards finish:

```bash
python scripts/merge_teacher_labels.py \
  --requests work/gigaspeech_pilot/compression_teacher_requests.jsonl \
  --labels \
    work/gigaspeech_pilot/compression_teacher_labels_shard0.jsonl \
    work/gigaspeech_pilot/compression_teacher_labels_shard1.jsonl \
  --output work/gigaspeech_pilot/compression_teacher_labels.jsonl \
  --sft-output work/gigaspeech_pilot/compression_sft.jsonl
```

## SFT Path

The current training entrypoint targets the first, safer milestone: tune the
text-side compression/translation behavior before touching audio generation.

```bash
python3 scripts/train_text_lora_sft.py --config configs/qwen3_omni_lora_sft.yaml --dry-run
```

On b200 the project venv was created with system site packages so it can reuse
the existing CUDA-enabled `torch`:

```bash
python3 -m pip install --user --break-system-packages -U virtualenv
python3 -m virtualenv --system-site-packages .venv
source .venv/bin/activate
pip install -U transformers datasets peft trl accelerate qwen-omni-utils \
  librosa soundfile audioread evaluate sacrebleu sentencepiece protobuf rich
```

The current choice is **Transformers Trainer + PEFT LoRA** using
`Qwen3OmniMoeProcessor` for the chat template and
`Qwen3OmniMoeThinkerForConditionalGeneration` for text/thinker model loading.
The top-level `Qwen3OmniMoeForConditionalGeneration` wrapper is used for
generation, but its `forward` path is not the right SFT training interface. The data collator
constructs completion-only labels, so loss is applied only to the compressed
assistant translation, not the system/user prompt.

Validate tokenization and label masking without loading model weights:

```bash
source .venv/bin/activate
python scripts/train_text_lora_sft.py \
  --config configs/qwen3_omni_lora_sft.yaml \
  --tokenize-smoke
```

Run a real LoRA SFT job after choosing idle GPUs:

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/train_text_lora_sft.py \
  --config configs/qwen3_omni_lora_sft.yaml
```

Pilot run on b200:

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/train_text_lora_sft.py \
  --config configs/qwen3_omni_lora_sft.yaml \
  --train-file work/gigaspeech_pilot/faithful_sft.jsonl \
  --output-dir runs/qwen3_omni_gigaspeech_faithful_warmup_pilot_20step \
  --max-steps 20 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4
```

This completed and saved a PEFT adapter at
`runs/qwen3_omni_gigaspeech_faithful_warmup_pilot_20step`.

## FLORAS Live S2S Benchmark

The FLORAS live benchmark scripts build a small long-form ZH<->EN closed-model
S2S eval. API keys are read only from environment variables and are never stored
in manifests or logs.

Prepare one long sample per direction and expand speeds `1.0,1.5,2.0`:

```bash
export OPENAI_API_KEY=...
export S2S_REFERENCE_MODEL=gpt-5-mini
python scripts/prepare_floras_live_manifest.py \
  --output-dir outputs/floras_live_pilot \
  --samples-per-direction 1 \
  --min-duration-s 600 \
  --speeds 1.0,1.5,2.0
```

By default the prepare script scans until it has a 32-sample candidate pool per
direction, then keeps the lowest-score samples. Set
`--stop-after-candidates-per-direction 0` to scan all reachable shards.
For `zh->en`, the script uses the FLORAS `translation` field when it is present.
If selected test rows have an empty `translation`, it generates the English
reference from the Chinese transcript with the configured text LLM. `en->zh`
always uses an LLM-generated Chinese reference because FLORAS is X-to-English.

Run a 10-second dry-run smoke without calling the live API:

```bash
python scripts/run_floras_openai_realtime.py \
  --manifest outputs/floras_live_pilot/live_runs.jsonl \
  --output-dir outputs/floras_live_pilot/live_smoke \
  --max-runs 1 \
  --smoke-seconds 10 \
  --dry-run

python scripts/evaluate_floras_live_s2s.py \
  --manifest outputs/floras_live_pilot/live_runs.jsonl \
  --run-output-dir outputs/floras_live_pilot/live_smoke \
  --output-dir outputs/floras_live_pilot/eval_smoke \
  --max-windows 1
```

Run the real OpenAI Realtime Translation pass with 960 ms paced input chunks:

```bash
export OPENAI_API_KEY=...
python scripts/run_floras_openai_realtime.py \
  --manifest outputs/floras_live_pilot/live_runs.jsonl \
  --output-dir outputs/floras_live_pilot/live_runs \
  --chunk-ms 960
```

Run the same benchmark through Gemini Live Translate. Gemini takes 16 kHz PCM
input and returns 24 kHz PCM output; the runner keeps the same result schema as
the OpenAI runner, so the ASR and HTML evaluation commands below can be reused.
The runner sends Gemini's audio-stream-end signal after the paced input and
keeps `generated_target_raw.wav` plus `audio_chunks_raw.jsonl` for the raw
server output. The default `generated_target.wav` and `audio_chunks.jsonl` trim
trailing silent PCM chunks so duration metrics track spoken target audio rather
than Gemini's silent tail frames.

```bash
export GEMINI_API_KEY=...
python scripts/run_floras_gemini_live.py \
  --manifest outputs/floras_live_pilot/live_runs.jsonl \
  --output-dir outputs/floras_live_pilot/gemini_live_runs \
  --chunk-ms 960
```

Then evaluate and render the UI:

```bash
python scripts/evaluate_floras_live_s2s.py \
  --manifest outputs/floras_live_pilot/live_runs.jsonl \
  --run-output-dir outputs/floras_live_pilot/live_runs \
  --output-dir outputs/floras_live_pilot/eval \
  --coverage-judge none
```

For ASR-based scoring of generated target speech, transcribe the generated wavs
first and pass the resulting JSONL into evaluation:

```bash
export OPENAI_API_KEY=...
python scripts/openai_transcribe_live_outputs.py \
  --run-output-dir outputs/floras_live_pilot/live_runs \
  --output outputs/floras_live_pilot/asr.jsonl

python scripts/evaluate_floras_live_s2s.py \
  --manifest outputs/floras_live_pilot/live_runs.jsonl \
  --run-output-dir outputs/floras_live_pilot/live_runs \
  --output-dir outputs/floras_live_pilot/eval_asr \
  --asr-jsonl outputs/floras_live_pilot/asr.jsonl \
  --coverage-judge none
```

The per-window HTML rows need per-window ASR if the transcript must match the
audio slice shown in that row. Generate it from an initial eval directory and
rerender. The page also shows a full-target-ASR context around each target
window; this is a wider approximate text context for manual omission checks,
not a narrow time-aligned transcript.

```bash
export OPENAI_API_KEY=...
python scripts/openai_transcribe_eval_windows.py \
  --eval-dir outputs/floras_live_pilot/eval_asr \
  --output outputs/floras_live_pilot/window_asr.jsonl

python scripts/evaluate_floras_live_s2s.py \
  --manifest outputs/floras_live_pilot/live_runs.jsonl \
  --run-output-dir outputs/floras_live_pilot/live_runs \
  --output-dir outputs/floras_live_pilot/eval_window_asr \
  --asr-jsonl outputs/floras_live_pilot/asr.jsonl \
  --window-asr-jsonl outputs/floras_live_pilot/window_asr.jsonl \
  --target-context-s 20 \
  --coverage-judge none
```
