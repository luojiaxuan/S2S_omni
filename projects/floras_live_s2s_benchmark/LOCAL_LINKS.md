# Local Audio And Dashboard Links

These links are for the original local machine. They point to complete HTML
dashboards with audio files present on disk. The wav files are intentionally not
tracked in Git. The current Seed AST chunk sweep has 1,308 local wav/window
files in the Git worktree, totaling about 721 MB, plus raw Seed run outputs
under the local artifact root.

## KIT Full-Source Debug Capture

This is not a formal KIT result. The 2026-07-06 run used the default
low-latency online configuration before the KIT settings were inspected and was
interrupted after 330.0s of the 1072.63s FLORAS source.

KIT session:

```text
https://lecture-translator.kit.edu/present/99976397598743707754867175416840867729
```

Local debug JSON:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_full_enzh/kit_live_enzh_full_realtime_run.json
```

Local partial dashboard generated only for extractor sanity checking:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_full_enzh/partial_dashboard_check/index.html
```

The cookie file in the same directory is local credential material and must not
be copied into Git. Do not rank this debug capture against full FLORAS results.

## KIT/GPT/Gemini/Seed 60s Compare

Same first 60s FLORAS EN->ZH source clip. GPT/Gemini rows use target-speech
ASR. KIT main rows use retrieved target speech scored through ASR; KIT
web-event TTS text rows are debug-only. Seed rows are marked as full-run prefix
proxies, not exact 60s Seed reruns. This is a smoke/debug dashboard, not a
formal KIT product score.

For the speed=1.5 dashboard, GPT/Gemini/Seed rows use local crops from the first
60s of existing full-run generated target wavs, then `gpt-4o-mini-transcribe`.
KIT uses the 60s source smoke run with `format=mixed` and target-speech ASR.

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark/artifacts/compare_gpt_gemini_seed_kit_enzh_60s/index.html
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark/artifacts/compare_gpt_gemini_seed_kit_enzh_60s_speed15/index.html
```

Local KIT target-audio and raw captures:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_upload_smoke/kit_live_enzh_run.json
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_upload_smoke/kit_live_enzh_realtime_run.json
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_upload_smoke/kit_extractor_diagnosis.json
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_config_smoke_60s_chunk1920/
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_speed15_60s_chunk1920/mixed_high_quality_no_post/
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/full_first60_target_asr/speed1p5/
```

## Seed AST Full Chunk/Speed Compare

Combined dashboard with OpenAI, Gemini, and Seed AST at 0.96s/1.92s chunks and
1.0x/1.5x speed:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark/artifacts/compare_openai_gemini_seed_enzh_full_chunks/index.html
```

## Seed AST Speed-1 Compare

Combined dashboard with OpenAI, Gemini, and Seed AST at 0.96s/1.92s chunks and
1.0x speed:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark/artifacts/compare_openai_gemini_seed_enzh_speed1/index.html
```

Seed AST 0.96s detail pages with per-window audio:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark/artifacts/eval_runs/seed_ast_chunk960_gpt4o_mini_asr/en-zh_mono_asr_test__0__speed_1/index.html
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark/artifacts/eval_runs/seed_ast_chunk960_gpt4o_mini_asr/en-zh_mono_asr_test__0__speed_1.5/index.html
```

Seed AST 1.92s detail pages with per-window audio:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark/artifacts/eval_runs/seed_ast_chunk1920_gpt4o_mini_asr/en-zh_mono_asr_test__0__speed_1/index.html
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark/artifacts/eval_runs/seed_ast_chunk1920_gpt4o_mini_asr/en-zh_mono_asr_test__0__speed_1.5/index.html
```

Seed AST raw run directories:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/seed_ast_full_enzh_chunk960
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/seed_ast_full_enzh_chunk1920
```

Seed AST target-speech ASR transcripts:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/seed_ast_asr_full_enzh_chunk960.jsonl
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/seed_ast_asr_full_enzh_chunk1920.jsonl
```

## OpenAI/Gemini Full Compare

Combined dashboard for OpenAI vs Gemini at 0.96s and 1.92s chunks:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/compare_openai_gemini_enzh_full_chunks/index.html
```

Full local artifact root:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs
```

## Git-Tracked Lightweight Artifacts

The corresponding JSON/HTML metadata tracked in Git is under:

```text
projects/floras_live_s2s_benchmark/artifacts/compare_openai_gemini_seed_enzh_speed1
projects/floras_live_s2s_benchmark/artifacts/compare_openai_gemini_seed_enzh_full_chunks
projects/floras_live_s2s_benchmark/artifacts/compare_gpt_gemini_seed_kit_enzh_60s
projects/floras_live_s2s_benchmark/artifacts/compare_gpt_gemini_seed_kit_enzh_60s_speed15
projects/floras_live_s2s_benchmark/artifacts/eval_runs/seed_ast_chunk960_gpt4o_mini_asr
projects/floras_live_s2s_benchmark/artifacts/eval_runs/seed_ast_chunk1920_gpt4o_mini_asr
projects/floras_live_s2s_benchmark/artifacts/compare_openai_gemini_enzh_full_chunks
```

The `*.wav` files are ignored by `.gitignore`; use the local paths above or
upload the audio bundle to Hugging Face / release assets for portable sharing.
