# FLORAS Live S2S Benchmark

This project bundle captures the current FLORAS live speech-to-speech benchmark
for EN->ZH long-form streaming translation.

## Scope

- Source data: FLORAS long-form EN monolingual ASR test sample.
- Backends: OpenAI Realtime, Gemini Live, Seed AST, and exploratory KIT
  Lecture Translator captures. The current full-source KIT rows use the
  corrected bilingual KIT source-language setting, while the older only-en KIT
  rows are retained only as historical diagnostics in local staging.
- Chunk sizes: 960 ms and 1920 ms.
- Speeds: 1.0x and 1.5x.
- Evaluation: target speech ASR, reference-based BLEU/chrF/CER, reference-free
  xCOMET-lite QE and MetricX-24 QE, window-level backlog, wall-clock playback
  delay, sentence/window coverage artifacts for manual inspection.

## Tracked Artifacts

- `RESULTS.md`: compact metric table.
- `LOCAL_LINKS.md`: local complete dashboard/audio paths for the original
  machine.
- `artifact_manifest.json`: copied artifact list plus large local audio pointers.
- `artifacts/compare_openai_gemini_enzh_full_chunks/index.html`: combined
  dashboard.
- `artifacts/compare_openai_gemini_enzh_full_chunks/compare_metrics.jsonl`:
  source-of-truth metric rows for the combined dashboard.
- `artifacts/compare_openai_gemini_seed_enzh_speed1/index.html`: one-sample
  dashboard comparing OpenAI, Gemini, and Seed AST at 1.0x speed.
- `artifacts/compare_openai_gemini_seed_enzh_speed1/compare_metrics.jsonl`:
  metric rows for the one-sample dashboard. Seed text is ASR over generated
  target speech, not the AST translation subtitle.
- `artifacts/compare_openai_gemini_seed_enzh_full_chunks/index.html`: full
  one-sample dashboard comparing OpenAI, Gemini, and Seed AST at 0.96s/1.92s
  chunks and 1.0x/1.5x speed.
- `artifacts/compare_openai_gemini_seed_enzh_full_chunks/compare_metrics.jsonl`:
  metric rows for the full Seed AST chunk/speed dashboard.
- `artifacts/compare_gpt_gemini_seed_kit_enzh_full/index.html`: full-source
  dashboard comparing GPT/Gemini/Seed/KIT at 1.0x and 1.5x. KIT now has
  0.96s and 1.92s chunk rows in this full dashboard. The active KIT rows use
  repeated `language=zh&language=en`, `mtLanguage=zh`, `audioLanguage=zh`,
  `format=mixed`, `ttsQualityMode=high_quality`, no postproduction parameter,
  and target-speech ASR.
- `artifacts/qe/full_enzh_qe_scores.jsonl`: per-full-run reference-free QE
  rows. xCOMET-QE uses `myyycroft/XCOMET-lite`; MetricX-QE uses
  `google/metricx-24-hybrid-large-v2p6-bfloat16`. Inputs are source transcript
  plus target-speech ASR hypothesis, split into proportional text chunks. The
  current QE file covers all 16 rows in the GPT/Gemini/Seed/KIT full dashboard.
  The current xCOMET-lite values are diagnostic/uncalibrated rather than
  calibrated 0-1 xCOMET scores; do not use them as an absolute quality score
  until the xCOMET inference path is fixed.
- `artifacts/qe/full_enzh_qe_segments.jsonl`,
  `artifacts/qe/full_enzh_xcomet_qe_segments.jsonl`, and
  `artifacts/qe/full_enzh_metricx_qe_segments.jsonl`: segment-level QE inputs
  and model outputs used to build the aggregate rows.
- `artifacts/compare_gpt_gemini_seed_kit_enzh_60s/index.html`: 60s smoke
  dashboard comparing OpenAI, Gemini, KIT Lecture Translator, and Seed AST
  proxy rows. This is a debug/tokenizer artifact, not the formal KIT product
  comparison.
- `artifacts/compare_gpt_gemini_seed_kit_enzh_60s/compare_metrics.jsonl`:
  metric rows for the 60s dashboard. Hypothesis/reference punctuation is
  preserved and both default-tokenizer BLEU and `tokenize=zh` BLEU are stored;
  the dashboard table only displays the `tokenize=zh` BLEU.
  The script verifies the exact 60s eval rows share the same
  `sentence_coverage.jsonl` reference. KIT main rows use retrieved target
  speech scored through ASR; KIT web-event TTS text rows are debug-only. Seed
  rows are prefix proxies from the full 1072s run, not exact 60s Seed reruns.
- `artifacts/compare_gpt_gemini_seed_kit_enzh_60s_speed15/index.html`:
  combined speed=1.0 and speed=1.5 dashboard over the same 60s EN->ZH source
  content. Speed=1.5 GPT/Gemini/Seed rows use the first 60s cropped from
  existing full-run generated target wavs and re-transcribed with
  `gpt-4o-mini-transcribe`; KIT uses the 60s source smoke with `format=mixed`,
  `ttsQualityMode=high_quality`, and 1.92s chunks.
- `artifacts/eval_runs/*`: per-backend/chunk `summary.json`, `metrics.jsonl`,
  `timeline.jsonl`, `sentence_coverage.jsonl`, and small HTML index files.
- `artifacts/root_metadata/*`: selected sample metadata, run manifest, ASR
  transcripts, and alignment helper JSONL files.

## Large Artifact Policy

The full run output under the local source directory is about 3.8 GB, mostly
full wav files and per-window wav slices. Those files are not tracked in Git.
The local source directory for this snapshot is:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs
```

The combined dashboard copied into this project still contains absolute links
to the local audio files so it remains usable on the original machine. For a
portable release, upload the audio bundle to Hugging Face or a GitHub release
asset and rewrite the dashboard links.

The Seed AST detail pages contain local wav/window references. Their
lightweight HTML/JSON metadata is tracked in Git, while the wavs remain local
and are ignored by `.gitignore`. Use `LOCAL_LINKS.md` for the full local
dashboard and audio paths.

KIT Lecture Translator target-speech retrieval is verified for smoke and full
sessions: `tts:0` linked audio chunks are resolved into target wavs and scored
through the same ASR path as GPT/Gemini/Seed. KIT web-event TTS text is still
shown only as debug text, because it can reflect product-side rewrite behavior
rather than the actual emitted target speech. The corrected 2026-07-06
full-source KIT run used repeated `language=zh&language=en`, `mtLanguage=zh`,
`audioLanguage=zh`, `format=mixed`, `ttsQualityMode=high_quality`, no
postproduction parameter, private availability, and 0.96s/1.92s input chunks:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_full_mixed_hq_chunk960_bilang_no_post
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_eval_full_mixed_hq_chunk960_bilang_no_post_asr
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_asr_full_mixed_hq_chunk960_bilang_no_post.jsonl
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_full_mixed_hq_chunk1920_bilang_no_post
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_eval_full_mixed_hq_chunk1920_bilang_no_post_asr
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_asr_full_mixed_hq_chunk1920_bilang_no_post.jsonl
```

At 0.96s chunks, KIT scored BLEU 18.32 / chrF 19.30 / CER 0.824 with xCOMET
0.0397 / MetricX-QE 9.101 at speed=1.0, and BLEU 18.92 / chrF 19.82 / CER
0.830 with xCOMET 0.0781 / MetricX-QE 10.323 at speed=1.5. At 1.92s chunks,
KIT scored BLEU 18.37 / chrF 19.12 / CER 0.827 with xCOMET 0.0284 /
MetricX-QE 8.277 at speed=1.0, and BLEU 18.90 / chrF 19.24 / CER 0.843 with
xCOMET 0.0389 / MetricX-QE 8.075 at speed=1.5. The 60s smoke advantage did not
carry over to the full wav; inspect the dashboard detail text and local audio
before treating KIT as competitive on the full sample. The 0.96s rows are
higher on the current diagnostic QE columns, especially at speed=1.5, but this
is a single-sample result and BLEU/CER do not show a clear quality win.

The earlier full-source mixed/high-quality KIT rows below are diagnostic only
because the session creation used `language=en` rather than the bilingual KIT
input-language setting:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_full_mixed_hq_chunk1920
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_eval_full_mixed_hq_chunk1920_asr
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_asr_full_mixed_hq_chunk1920.jsonl
```

Follow-up 60s KIT configuration checks on 2026-07-06 showed that this was not
just an HTML label issue. `scripts/create_kit_session.py` now supports repeated
`--language` arguments and encodes them as repeated query parameters. No-post
`format=mixed`, `ttsQualityMode=high_quality`, `audioLanguage=zh` smokes scored
as follows on the same first-60s FLORAS clip:

```text
language=en&language=zh: BLEU 21.74, chrF 20.69, CER 0.758, target 69.95s, 13 TTS audio chunks
language=zh&language=en: BLEU 20.96, chrF 20.55, CER 0.775, target 72.63s, 14 TTS audio chunks
```

The current saved KIT `profile_1` expands to `language=zh,en`,
`mtLanguage=zh,en`, `audioLanguage=zh`, `format=mixed`,
`ttsQualityMode=high_quality`, and `postproduction=50` with the permanent
shorten name `siqiouyaandrewcmuedu`. That profile is not the no-compression S2S
main setting; the 60s target speech was 247.20s and scored BLEU 7.12 / chrF
15.16 / CER 3.517 against the 60s reference.

The 2026-07-06 full-source KIT attempt used a default low-latency online
configuration and was interrupted after 330 seconds of the 1072.63-second source
because the configuration had not been optimized. Treat that capture as local
debug data only.

## Metric Definitions

- `duration_lag_s`: generated target wav duration minus streamed source wav
  duration.
- `wall_delay_s`: simulated wall-clock time when target playback finishes minus
  source stream end time.
- `max_backlog_s`: maximum window-level emitted-target deficit during streaming.
- `max_playback_queue_s`: maximum target-audio queue ahead of the live source.
- `BLEU`, `chrF`, `CER`: reference-based metrics computed from target-speech
  ASR transcript against the current GPT-generated target reference text.
- `xCOMET-QE`: reference-free quality estimate from
  `myyycroft/XCOMET-lite`, using source transcript plus target-speech ASR
  hypothesis only. The current run is diagnostic/uncalibrated: segment scores
  include negative values and aggregates are near zero, so this should not be
  read as a calibrated 0-1 xCOMET quality score.
- `MetricX-QE`: reference-free score derived from
  `google/metricx-24-hybrid-large-v2p6-bfloat16`, using source transcript plus
  target-speech ASR hypothesis only. The model's raw `MetricX err` is
  lower-is-better on a 0-25 scale; the dashboard reports `25 - err` as
  higher-is-better `MetricX-QE` and keeps the raw error column.
- QE is computed on proportional text chunks because the full 1072s transcript
  is too long for these learned metrics. This is an approximate document-level
  segmentation, not a time-aligned or sentence-aligned comparison; interpret QE
  carefully for high-backlog, truncated, or heavily compressed runs.

## Refresh Command

From the repository root:

```bash
python3 scripts/package_floras_live_project.py \
  --source-dir /Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs \
  --project-dir /Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark
```

Seed AST live outputs are produced by:

```text
scripts/run_floras_seed_ast.py
```

The KIT/GPT/Gemini/Seed 60s smoke dashboard is rebuilt with:

```bash
python3 scripts/build_floras_kit_60s_compare.py \
  --source-root /Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs \
  --project-dir /Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark \
  --run-id en-zh_mono_asr_test__0__speed_1 \
  --output-name compare_gpt_gemini_seed_kit_enzh_60s \
  --sacrebleu-path /path/to/sacrebleu/site-packages
```

The combined speed=1.0/1.5 dashboard is rebuilt by changing `--run-id` to
`en-zh_mono_asr_test__0__speed_1.5`, adding
`--include-run-id en-zh_mono_asr_test__0__speed_1`, and setting
`--output-name compare_gpt_gemini_seed_kit_enzh_60s_speed15`. For the speed=1.5
run id, the script loads local `full_first60_target_asr/speed1p5/*/target_first60.wav`
crops for GPT/Gemini/Seed when present; those wav/ASR artifacts are local
staging files and are intentionally not committed.

Reference-free QE inputs are built from the full KIT/GPT/Gemini/Seed
`compare_metrics.jsonl`, then scored with xCOMET-lite and MetricX-24:

```bash
python3 scripts/build_floras_qe_inputs.py \
  --manifest projects/floras_live_s2s_benchmark/artifacts/root_metadata/live_runs.jsonl \
  --compare-metrics projects/floras_live_s2s_benchmark/artifacts/compare_gpt_gemini_seed_kit_enzh_full/compare_metrics.jsonl \
  --output-segments projects/floras_live_s2s_benchmark/artifacts/qe/full_enzh_qe_segments.jsonl \
  --output-runs projects/floras_live_s2s_benchmark/artifacts/qe/full_enzh_qe_runs.jsonl

python3 scripts/run_floras_qe_xcomet.py \
  --input-jsonl projects/floras_live_s2s_benchmark/artifacts/qe/full_enzh_qe_segments.jsonl \
  --output-jsonl projects/floras_live_s2s_benchmark/artifacts/qe/full_enzh_xcomet_qe_segments.jsonl \
  --model-name myyycroft/XCOMET-lite \
  --xcomet-code-dir /path/to/xCOMET-lite

python3 scripts/run_floras_qe_metricx.py \
  --input-jsonl projects/floras_live_s2s_benchmark/artifacts/qe/full_enzh_qe_segments.jsonl \
  --output-jsonl projects/floras_live_s2s_benchmark/artifacts/qe/full_enzh_metricx_qe_segments.jsonl \
  --model-name google/metricx-24-hybrid-large-v2p6-bfloat16 \
  --tokenizer google/mt5-large \
  --metricx-code-dir /path/to/metricx

python3 scripts/aggregate_floras_qe_scores.py \
  --segments-jsonl projects/floras_live_s2s_benchmark/artifacts/qe/full_enzh_qe_segments.jsonl \
  --runs-jsonl projects/floras_live_s2s_benchmark/artifacts/qe/full_enzh_qe_runs.jsonl \
  --xcomet-jsonl projects/floras_live_s2s_benchmark/artifacts/qe/full_enzh_xcomet_qe_segments.jsonl \
  --metricx-jsonl projects/floras_live_s2s_benchmark/artifacts/qe/full_enzh_metricx_qe_segments.jsonl \
  --output-jsonl projects/floras_live_s2s_benchmark/artifacts/qe/full_enzh_qe_scores.jsonl
```

Then rebuild the full-source dashboard with:

```bash
python3 scripts/build_floras_kit_full_compare.py \
  --manifest /Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark/artifacts/root_metadata/live_runs.jsonl \
  --output-dir /Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark/artifacts/compare_gpt_gemini_seed_kit_enzh_full \
  --run-id-prefix en-zh_mono_asr_test__0__speed_ \
  --qe-scores-jsonl /Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark/artifacts/qe/full_enzh_qe_scores.jsonl \
  --require-qe \
  --eval openai_960=/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/openai_eval_full_enzh_chunk960_asr \
  --eval openai_1920=/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/openai_eval_full_enzh_chunk1920_asr \
  --eval gemini_960=/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/gemini_eval_full_enzh_chunk960_trim_asr \
  --eval gemini_1920=/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/gemini_eval_full_enzh_chunk1920_trim_asr \
  --eval seed_960=/Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark/artifacts/eval_runs/seed_ast_chunk960_gpt4o_mini_asr \
  --eval seed_1920=/Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark/artifacts/eval_runs/seed_ast_chunk1920_gpt4o_mini_asr \
  --eval kit_960_bilang_no_post=/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_eval_full_mixed_hq_chunk960_bilang_no_post_asr \
  --eval kit_1920_bilang_no_post=/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_eval_full_mixed_hq_chunk1920_bilang_no_post_asr
```

## Current Takeaway

The dashboard separates duration-level lag from wall-clock delay. For example,
Gemini at 960 ms chunks and 1.0x source speed has only 5.62 s duration lag, but
185.97 s wall-clock delay and 114.00 s max backlog, meaning the final audio
length is close to the source but the live system returned audio far too late.
For the 60s KIT smoke comparison, the earlier BLEU 0.0 reading was an eval
artifact: the same hypotheses score 0.00 with sacreBLEU's default tokenizer but
20-26 BLEU with `tokenize=zh`, while preserving the original
hypothesis/reference punctuation. Do not use KIT web-event TTS text as the main
S2S hypothesis. `format=mixed` is acceptable when the hypothesis comes from the
retrieved target speech and is scored through ASR. The speed=1.5 comparison now
uses full-run target-wav first-60s crops for GPT/Gemini/Seed, so those rows are
not exact 60s source replays; the crop can include content beyond the first-60s
reference and should be interpreted as the requested target-audio crop view.
Seed crop rows are especially vulnerable to this windowing artifact and should
not be read as a source-time-aligned quality ranking.
The current full-source KIT rows are corrected bilingual/no-post runs:
`language=zh&language=en`, `mtLanguage=zh`, `audioLanguage=zh`, `format=mixed`,
`ttsQualityMode=high_quality`, and target-speech ASR, now at both 0.96s and
1.92s input chunks. The earlier 60s smoke advantage did not hold on the full
wav. QE has been rerun for all 16 dashboard rows. The current diagnostic QE
columns are higher for KIT 0.96s than 1.92s, especially at speed=1.5, but
xCOMET is uncalibrated here and BLEU remains around 18-19.
