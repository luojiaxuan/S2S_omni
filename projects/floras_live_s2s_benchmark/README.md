# FLORAS Live S2S Benchmark

This project bundle captures the current FLORAS live speech-to-speech benchmark
for EN->ZH long-form streaming translation.

## Scope

- Source data: FLORAS long-form EN monolingual ASR test sample.
- Backends: OpenAI Realtime, Gemini Live, and Seed AST for source-to-speech
  comparison.
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

## Current Takeaway

The dashboard separates duration-level lag from wall-clock delay. For example,
Gemini at 960 ms chunks and 1.0x source speed has only 5.62 s duration lag, but
185.97 s wall-clock delay and 114.00 s max backlog, meaning the final audio
length is close to the source but the live system returned audio far too late.
