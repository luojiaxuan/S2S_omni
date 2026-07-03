# Local Audio And Dashboard Links

These links are for the original local machine. They point to complete HTML
dashboards with audio files present on disk. The wav files are intentionally not
tracked in Git because the Seed AST detail run contains 327 wav files totaling
about 200 MB.

## Seed AST Speed-1 Compare

Combined dashboard with OpenAI, Gemini, and Seed AST at 1.0x:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark/artifacts/compare_openai_gemini_seed_enzh_speed1/index.html
```

Seed AST detail page with per-window audio:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark/artifacts/eval_runs/seed_ast_gpt4o_mini_asr_speed1/en-zh_mono_asr_test__0__speed_1/index.html
```

Seed AST full/audio window directory:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni/projects/floras_live_s2s_benchmark/artifacts/eval_runs/seed_ast_gpt4o_mini_asr_speed1/en-zh_mono_asr_test__0__speed_1
```

Seed AST chunk-size follow-up status:

```text
Requested: chunk_ms=960 and chunk_ms=1920 for speed=1.0 and speed=1.5.
Current local credential status: blocked by Seed AST service error
"quota exceeded for types: tokens_lifetime".
Completed Seed AST result currently available: chunk_ms=100, speed=1.0.
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
projects/floras_live_s2s_benchmark/artifacts/eval_runs/seed_ast_gpt4o_mini_asr_speed1
projects/floras_live_s2s_benchmark/artifacts/compare_openai_gemini_enzh_full_chunks
```

The `*.wav` files are ignored by `.gitignore`; use the local paths above or
upload the audio bundle to Hugging Face / release assets for portable sharing.
