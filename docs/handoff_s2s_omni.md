# S2S_omni Handoff

Last updated: 2026-07-04

Repository:

```text
https://github.com/luojiaxuan/S2S_omni
```

Current local checkout:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni
```

## Context

This project studies streaming speech-to-speech translation when the source
speaker is dense or fast enough that a faithful target translation cannot finish
playing before the next source chunk arrives.

The target problem is not simply "source speech is fast." The core problem is:

```text
How should a live S2S system shorten target content, preserve core meaning, and
avoid growing playback backlog, without making the target speech unnatural or
hard to listen to?
```

The main research angle is the tradeoff among:

- semantic fidelity to the reference or source meaning
- audio naturalness and listenability
- live feasibility, measured as target audio being available and playable in
  wall-clock time
- controlled compression instead of silent sentence dropping

The initial base model focus was Qwen3-Omni. The project later added closed
model live benchmarks and a Seed / ByteDance AST metrics script because model
training alone was not enough to understand the live backlog problem.

## What Exists In The Repo

The repo contains four broad categories of work.

1. Compression and SFT data infrastructure:

- GigaSpeech policy data builders.
- RTF-aware pass-through vs compression decision logic.
- Teacher label generation, merge, audit, and split verification.
- Text-side LoRA SFT entrypoints for Qwen3-Omni.

2. Speech / talker training experiments:

- Qwen3-Omni talker/code capture helpers.
- Soft-wav / differentiable target audio training experiments.
- Wav2codec experiments and related audit scripts.
- RASST / Qwen3-Omni soft-wav manifest and training runners.

3. Evaluation and benchmark infrastructure:

- Text metrics: BLEU, chrF, CER/WER through ASR, bag-F1, length ratios.
- Streaming metrics: S2S RTF, duration lag, window-level backlog, wall-clock
  playback delay.
- FLORAS live S2S benchmark for OpenAI Realtime and Gemini Live.
- HTML inspection pages for human listening and debugging.

4. Project bundles:

- `projects/floras_live_s2s_benchmark/`
- `projects/acl6060_s2s_metrics_seed/`

These bundles are intended to preserve concrete experiment outputs or imported
external scripts in a reviewable Git form.

## Important Lessons So Far

### Compression Data

The compression decision should be based on target speech feasibility, not
source speed alone.

Current policy:

```text
faithful_target_speech_s = target_units / default_target_unit_rate
source_chunk_wall_s = source_duration_s / source_speed_factor
s2s_rtf = faithful_target_speech_s / source_chunk_wall_s
```

If `s2s_rtf <= 1.0`, keep a faithful/pass-through target. If `s2s_rtf > 1.0`,
ask the teacher to compact the target text under a duration-derived character or
unit budget.

The full GigaSpeech scan showed the natural split is already close to balanced:
about 50.2% pass-through and 49.8% compression across speed factors
`1.0,1.35,1.7,2.0`. Do not manually force the ratio unless running an ablation.

### Split Discipline

Formal evaluation must split by original `base_id`, not by stressed variant id.
All variants such as `AUD...__speed_1.7` and `AUD...__speed_2` inherit the same
train/dev/test split as their original sample.

Earlier quick checks had train/eval overlap and should not be reported as held
out results.

### Thinker-Only LoRA Is Not Enough

The 25k thinker-only Qwen3-Omni LoRA learned text-side compression behavior and
improved held-out text metrics. However, using the compressed text with a frozen
talker did not reliably shorten generated speech duration.

Key observation:

```text
Text length can shrink while generated target wav duration does not.
```

This means the audio/talker path needs direct duration supervision or a cascade
TTS architecture. Text-only SFT is useful for policy learning, but it is not a
complete S2S backlog solution.

### Wav2codec Is Not A Simple Engineering Fix

Trying to train an Omni-compatible wav->codec inverter is problematic because
Qwen3-Omni talker codes behave like generated latent control tokens, not a
standard invertible neural audio codec. The code2wav direction can be many-to-one
and may include information not recoverable from the final wav.

Practical consequence:

```text
Do not assume TTS wav -> Omni codes can be learned with simple CE over frozen
talker codes.
```

The more promising Qwen3-Omni-native route is end-to-end training with frozen
code2wav and differentiable soft/st-argmax code selection. Early pilots showed
this can affect duration, but it is not yet a final robust result.

### Live Benchmark Metrics Need Wall-Clock Delay

Duration lag alone is misleading.

Example from FLORAS Gemini 960 ms chunk, 1.0x speed:

```text
source stream duration: 1072.63s
target audio duration: 1078.25s
duration lag: 5.62s
wall-clock playback delay: 185.97s
max backlog: 114.00s
```

The target audio duration is close to the source duration, but the backend
returns audio late. The user experience is still bad because the listener waits
for output.

The key live metrics are:

- `duration_lag_s`: target wav duration minus source stream duration.
- `wall_clock_end_delay_s`: simulated target playback end wall-clock time minus
  source stream end time.
- `max_backlog_s`: maximum window-level emitted-target deficit during streaming.

## Key Directories

```text
configs/
  Training and policy config templates.

docs/
  eval_plan.md
  hibiki_zero_backlog_route.md
  remote_artifacts.md
  handoff_s2s_omni.md

s2s_omni/
  Core Python helpers for data schema, metrics, prompts, streaming transforms,
  audio helpers, TTS abstractions, FLORAS live eval utilities, etc.

scripts/
  Dataset builders, teacher generation, SFT runners, live benchmark runners,
  ASR/eval/rendering scripts, Docker/remote helpers.

projects/
  Reviewable project bundles for concrete experiments or imported scripts.
```

## Project Bundles

### FLORAS Live S2S Benchmark

Path:

```text
projects/floras_live_s2s_benchmark/
```

Purpose:

- Preserve the current OpenAI Realtime vs Gemini Live EN->ZH benchmark.
- Track metrics and HTML dashboards without committing multi-GB wav artifacts.

Important files:

```text
projects/floras_live_s2s_benchmark/README.md
projects/floras_live_s2s_benchmark/LOCAL_LINKS.md
projects/floras_live_s2s_benchmark/RESULTS.md
projects/floras_live_s2s_benchmark/artifacts/compare_openai_gemini_enzh_full_chunks/index.html
projects/floras_live_s2s_benchmark/artifacts/compare_openai_gemini_enzh_full_chunks/compare_metrics.jsonl
projects/floras_live_s2s_benchmark/artifacts/compare_openai_gemini_seed_enzh_full_chunks/index.html
projects/floras_live_s2s_benchmark/artifacts/compare_openai_gemini_seed_enzh_full_chunks/compare_metrics.jsonl
```

Large wav artifacts are local only:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs
```

If this benchmark needs to be shared beyond the original machine, upload the
audio bundle to Hugging Face or a GitHub release asset and rewrite dashboard
links.

The Seed AST chunk/speed sweep is packaged here. It covers 0.96s and 1.92s
chunks at source speeds 1.0x and 1.5x, evaluated by running
`gpt-4o-mini-transcribe` over the generated target speech. The AST backend
translation subtitle is not used for BLEU/chrF/CER. The tracked HTML/JSON lives
under:

```text
projects/floras_live_s2s_benchmark/artifacts/compare_openai_gemini_seed_enzh_speed1
projects/floras_live_s2s_benchmark/artifacts/compare_openai_gemini_seed_enzh_full_chunks
projects/floras_live_s2s_benchmark/artifacts/eval_runs/seed_ast_chunk960_gpt4o_mini_asr
projects/floras_live_s2s_benchmark/artifacts/eval_runs/seed_ast_chunk1920_gpt4o_mini_asr
```

The Seed AST detail pages reference local wav/window files that are not tracked
in Git. Use this file for the exact local dashboard, detail, raw run, and ASR
transcript paths:

```text
projects/floras_live_s2s_benchmark/LOCAL_LINKS.md
```

Seed AST summary on the selected FLORAS example:

```text
chunk=960,  speed=1.0: BLEU 21.48, chrF 21.85, CER 0.836, wall delay 1.75s, max backlog 314.53s
chunk=1920, speed=1.0: BLEU 20.81, chrF 21.53, CER 0.812, wall delay 1.58s, max backlog 288.37s
chunk=960,  speed=1.5: BLEU 21.13, chrF 21.65, CER 0.805, wall delay 24.40s, max backlog 241.92s
chunk=1920, speed=1.5: BLEU 21.30, chrF 21.46, CER 0.818, wall delay 2.13s, max backlog 203.37s
```

### ACL6060 / Seed AST S2S Metrics Script

Path:

```text
projects/acl6060_s2s_metrics_seed/
```

Purpose:

- Track a Seed / ByteDance AST S2S evaluation script and required protobufs.
- The script streams wav input to AST S2S and writes target wav, timeline JSON,
  and transcript text.

Important files:

```text
projects/acl6060_s2s_metrics_seed/README.md
projects/acl6060_s2s_metrics_seed/vendor/seed/generate.py
projects/acl6060_s2s_metrics_seed/vendor/seed/protos/
projects/acl6060_s2s_metrics_seed/vendor/seed/python_protogen/
```

The downloaded script originally contained hard-coded credentials. Those were
removed before committing to the public repo. Credentials must be passed
explicitly via CLI arguments.

2026-07-04 ACL6060 EN->ZH dev reproduction correction:

The correct ACL6060 main-result input is streaming over 5 full wavs from the
RASST release data, not 468 independent segmented sentence wavs. The release
data is:

```text
https://huggingface.co/datasets/gavinlaw/rasst-main-result-data
```

Use the new full-wav streaming runner:

```text
projects/acl6060_s2s_metrics_seed/run_acl6060_live_stream_eval.py
```

It downloads/reads:

```text
main_result/audio/acl6060/2022.acl-long.{110,117,268,367,590}.wav
main_result/inputs/acl_zh/source.list
main_result/inputs/acl_zh/target.list
main_result/inputs/acl_zh/ref.txt
main_result/inputs/acl_zh/source_text.txt
main_result/inputs/acl_zh/audio.yaml
```

and writes a 5-row RASST-style `instances.log` where each row corresponds to a
full talk wav. `offline_streamlaal_eval.py` is still the right scorer; despite
the name, it reads a pre-generated streaming log and computes StreamLAAL/BLEU.

Local validation on 2026-07-04:

```text
/tmp/rasst_main_result_data        # HF ACL6060 subset, 107 MB
/tmp/acl6060_stream_dry_run        # dry-run log structure check
/tmp/acl6060_stream_openai_smoke_real_fixed  # 15s real OpenAI smoke
/tmp/acl6060_stream_gemini_smoke_real        # 15s real Gemini smoke
/tmp/acl6060_stream_gemini_split_smoke_real  # 6s Gemini split-session smoke
```

The 5 full wavs total about 57.4 minutes. A paced OpenAI/Gemini live run should
therefore take about an hour; `--no-pace` is only a fast protocol/debug mode.
OpenAI and Gemini 15-second real API smokes produced Chinese target transcript
deltas with 0 API errors, so the key files and live API paths are valid. Gemini
full runs should use `--max-session-input-s 480`, matching the FLORAS live
runner's service-sized session split.

Previous segmented diagnostic, kept only for tokenizer/scorer context:

```text
old Infinisst/RAG zh log: BLEU 51.316, LAAL 6639.597, AP 1.034
OpenAI gpt-audio-mini:  BLEU 51.438, LAAL 6639.597, AP 1.668
Gemini 3.5 Flash:       BLEU 50.106, LAAL 6639.597, AP 0.958
```

Result artifacts:

```text
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_dev_enzh_compare/
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_dev_enzh_openai_gpt_audio_mini/
projects/acl6060_s2s_metrics_seed/artifacts/acl6060_dev_enzh_gemini_audio/
```

Those 468-row artifacts are not valid evidence for streaming input because they
use independent segmented wav API calls and end-of-sentence delays. They are
still useful for the tokenizer question: ACL6060 zh BLEU is text prediction vs
text reference with `sacrebleu-tokenizer=zh`, not target-speech ASR BLEU. The
same old log scores 51.316 with tokenizer `zh` but only 3.479 with default
`13a`.

## Major Remote Artifacts

See `docs/remote_artifacts.md` for the full list. The most important entries
are summarized here.

### GigaSpeech Inputs

Original TSV on Taurus:

```text
/mnt/taurus/data/siqiouyang/datasets/gigaspeech/train_xl_case_ft-qwen2.5-32b-instruct_marked_mfa_punc_asr.tsv
```

MFA resources:

```text
/mnt/taurus/data/siqiouyang/datasets/gigaspeech/textgrids
/mnt/gemini/data1/jiaxuanluo/gigaspeech_mfa_index/gigaspeech_mfa_index.sqlite
```

### Natural-Policy 25k Dataset

B200:

```text
/data/repo/S2S_omni/work/gigaspeech_policy_pool_30k_lazy_20260622
```

Mirrored on Taurus:

```text
/mnt/data2/jiaxuanluo/S2S_omni/work/gigaspeech_policy_pool_30k_lazy_20260622
```

Key files:

```text
sft_25k.jsonl
manifest_25k.jsonl
tts_requests_25k.jsonl
sft_25k_summary.json
sft_25k_rejected.jsonl
```

Dataset summary:

- 25k records.
- 13,807 pass-through examples.
- 11,193 compression examples.
- 0 target-character-budget violations in final audit.
- 0 estimated-duration-budget violations in final audit.
- 0 style-guard violations in final audit.

### Final 25k Thinker-LoRA Run

Checkpoint:

```text
/data/checkpoints/s2s_omni/qwen3_omni_25k_thinker_lora_20260622_full
```

Held-out eval:

```text
/data/outputs/s2s_eval_qwen3_omni_25k_thinker_lora_20260622_full
```

Local synced eval copy:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/s2s_eval_qwen3_omni_25k_thinker_lora_20260622_full
```

Key held-out result:

```text
dev  base chrF 23.50, evo chrF 34.22
test base chrF 22.17, evo chrF 33.65
```

But generated wav duration did not reliably shrink with thinker-only LoRA, so
this is not a complete S2S duration-control solution.

## Important Scripts

Data and compression policy:

```text
scripts/build_gigaspeech_sft.py
scripts/assemble_policy_sft_dataset.py
scripts/generate_teacher_labels.py
scripts/merge_teacher_labels.py
scripts/verify_split_integrity.py
```

Qwen3-Omni / soft-wav / talker training:

```text
scripts/train_text_lora_sft.py
scripts/train_qwen3_omni_softwav_lora.py
scripts/build_rasst_softwav_manifest.py
scripts/generate_rasst_softwav_outputs.py
scripts/run_s2sonly_softwav_12k_aries.sh
scripts/run_rasst_softwav_full_aries.sh
```

FLORAS live benchmark:

```text
scripts/prepare_floras_live_manifest.py
scripts/run_floras_openai_realtime.py
scripts/run_floras_gemini_live.py
scripts/run_floras_seed_ast.py
scripts/openai_transcribe_live_outputs.py
scripts/openai_transcribe_eval_windows.py
scripts/floras_target_mfa.py
scripts/evaluate_floras_live_s2s.py
scripts/render_floras_compare_dashboard.py
scripts/package_floras_live_project.py
```

Hibiki-Zero / cascade route:

```text
scripts/hibiki_zero_prepare_sources.py
scripts/hibiki_zero_generate_teacher_text.py
scripts/hibiki_zero_generate_tts_targets.py
scripts/hibiki_zero_slice_mfa_chunks.py
scripts/hibiki_zero_run_baseline.py
```

## Compute Context

Use the shared-machine rules in `AGENTS.md`.

Practical convention:

- Taurus / Aries A6000: light data processing, smoke tests, sample eval,
  FLORAS/ASR/MFA-style work.
- B200: large Qwen3-Omni training, high-memory model work, RL.
- Keep code, caches, logs, checkpoints, and generated artifacts under a
  persistent personal data directory.
- Do not treat local `/data` or `/mnt/data*` paths as canonical source of truth
  for reusable datasets or models. Upload reusable artifacts to Hugging Face or
  record their intended HF destination.

Known project mirrors:

```text
B200:   /data/repo/S2S_omni
Taurus: /mnt/data2/jiaxuanluo/S2S_omni
Local:  /Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni
```

## Security Notes

- The repo is public.
- Do not commit API keys.
- The imported Seed AST script had hard-coded keys in the downloaded source.
  They were removed in commit `48a9285`.
- FLORAS live benchmark scripts read OpenAI/Gemini credentials at runtime. Keep
  keys in local runtime config, not in repo artifacts.
- Run a quick secret scan before committing any new vendor script or live API
  output.

## Recommended Next Steps

1. Decide the next training route.

   The strongest near-term options are:

   - Continue Qwen3-Omni end-to-end soft-wav/talker LoRA experiments if the goal
     is a single-model S2S result.
   - Use a cascade architecture if the goal is faster reliable backlog reduction:
     speech->compressed text policy plus a strong voice-copy TTS backend.

2. Integrate Seed AST into the FLORAS-style benchmark runner.

   The imported script already produces target wav, timeline JSON, and transcript
   text. The next useful step is to normalize its output into the same metrics
   schema as OpenAI/Gemini:

   ```text
   generated_target.wav
   audio_chunks.jsonl or equivalent timeline
   ASR transcript
   metrics.jsonl
   per-window HTML dashboard
   ```

3. Finish portable artifact storage.

   The FLORAS project bundle keeps lightweight JSON/HTML in Git, but the wavs
   are still local. For collaboration, package large audio artifacts as a HF
   dataset or release asset.

4. Expand live benchmarks beyond one EN->ZH sample.

   The first packaged benchmark covers one long EN->ZH sample over two chunk
   sizes and two speed settings. Add ZH->EN and more FLORAS samples before
   drawing benchmark-level conclusions.

5. Keep eval semantics clear.

   Always report wall-clock delay and max backlog, not only duration lag. For
   semantic quality, BLEU/chrF/CER are useful but insufficient; use human
   listening and LLM-as-judge for missed or compressed sentence judgments.

## Quick Orientation Commands

```bash
cd /Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni
git status --short
git log --oneline -10

python3 -m py_compile \
  scripts/evaluate_floras_live_s2s.py \
  scripts/render_floras_compare_dashboard.py \
  scripts/package_floras_live_project.py \
  projects/acl6060_s2s_metrics_seed/vendor/seed/generate.py

sed -n '1,120p' projects/floras_live_s2s_benchmark/RESULTS.md
sed -n '1,120p' docs/remote_artifacts.md
```

## Current Repository State At Handoff

Recent relevant commits:

```text
48a9285 Add ACL6060 Seed S2S metrics script
46bec80 Package FLORAS live benchmark project
27dbf1d Add wall clock delay to FLORAS dashboard
16bd9c4 Show overlapping gold sentences in FLORAS windows
14f6d01 Use window ASR transcript for FLORAS eval
c368c26 Add FLORAS target MFA alignment workflow
```

The repository is public and currently tracks code, docs, lightweight benchmark
artifacts, and vendor reference scripts. Large generated wavs, model checkpoints,
and training datasets are intentionally outside Git.
