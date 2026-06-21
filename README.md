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

## GigaSpeech Data

The first data source is the existing GigaSpeech S2TT table on taurus:

```text
/mnt/taurus/data/siqiouyang/datasets/gigaspeech/train_xl_case_ft-qwen2.5-32b-instruct_marked_mfa_punc_asr.tsv
```

The builder first splits by original GigaSpeech `base_id`, then emits two kinds
of records inside each split:

- `faithful_sft.jsonl`: phase-0 warm-up, source transcript to full translation
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
  --teacher-limit 2000 \
  --dev-teacher-limit 300 \
  --test-teacher-limit 300 \
  --split-ratios train=0.9,dev=0.05,test=0.05 \
  --out-dir work/gigaspeech_pilot_split
```

The split pilot scanned 100,000 rows and found 98,990 usable rows. It wrote
2,000 train teacher requests, 300 dev teacher requests, 300 test teacher
requests, and 4,000 train faithful SFT records. The split unit is `base_id`, so
different speed variants of the same utterance cannot cross train/dev/test.

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
