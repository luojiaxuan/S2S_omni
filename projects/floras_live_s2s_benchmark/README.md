# FLORAS Live S2S Benchmark

This project bundle captures the current FLORAS live speech-to-speech benchmark
for EN->ZH long-form streaming translation.

## Scope

- Source data: FLORAS long-form EN monolingual ASR test sample.
- Backends: OpenAI Realtime, Gemini Live, Seed AST, and exploratory KIT
  Lecture Translator captures. KIT now has a full-source mixed/high-quality
  target-speech-ASR comparison, but it is still a single product setting rather
  than a full KIT configuration sweep.
- Chunk sizes: 960 ms and 1920 ms.
- Speeds: 1.0x and 1.5x.
- Evaluation: target speech ASR, BLEU/chrF/CER, window-level backlog, wall-clock
  playback delay, sentence/window coverage artifacts for manual inspection.

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
  dashboard comparing GPT/Gemini/Seed/KIT at 1.0x and 1.5x. KIT currently has
  only the 1.92s chunk row in this full dashboard. KIT uses
  `format=mixed`, `ttsQualityMode=high_quality`, 1.92s input chunks, retrieved
  target speech, and `gpt-4o-mini-transcribe`.
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
rather than the actual emitted target speech. The 2026-07-06 full-source
mixed/high-quality KIT runs are local staging artifacts under:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_full_mixed_hq_chunk1920
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_eval_full_mixed_hq_chunk1920_asr
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_asr_full_mixed_hq_chunk1920.jsonl
```

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
- `BLEU`, `chrF`, `CER`: computed from ASR transcript against the target
  reference text.

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

The full-source KIT comparison dashboard is rebuilt after running KIT and ASR
with:

```bash
python3 scripts/build_floras_kit_full_compare.py \
  --manifest /Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/live_runs.jsonl \
  --output-dir /Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark/artifacts/compare_gpt_gemini_seed_kit_enzh_full \
  --run-id-prefix en-zh_mono_asr_test__0__speed_ \
  --eval openai_960=/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/openai_eval_full_enzh_chunk960_asr \
  --eval openai_1920=/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/openai_eval_full_enzh_chunk1920_asr \
  --eval gemini_960=/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/gemini_eval_full_enzh_chunk960_trim_asr \
  --eval gemini_1920=/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/gemini_eval_full_enzh_chunk1920_trim_asr \
  --eval seed_960=/Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark/artifacts/eval_runs/seed_ast_chunk960_gpt4o_mini_asr \
  --eval seed_1920=/Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark/artifacts/eval_runs/seed_ast_chunk1920_gpt4o_mini_asr \
  --eval kit_1920=/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_eval_full_mixed_hq_chunk1920_asr
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
The full-source KIT mixed/high-quality run is more comparable than the 60s
smoke/crop dashboard. On this sample it scores below the best GPT/Gemini/Seed
full-run rows: KIT 1.92s gets BLEU 18.29 at speed=1.0 and BLEU 17.46 at
speed=1.5, with target wavs about 107-113 seconds shorter than the streamed
source audio. The same evaluator computes KIT timing metrics from retrieved
`tts:0` audio chunk arrival times. Treat this as an exploratory single-setting
KIT result, not a complete KIT product sweep.
