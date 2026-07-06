## Full 1072s Sample

| run | backend | chunk_ms | speed | BLEU | chrF | CER | duration_lag_s | wall_delay_s | max_backlog_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| en-zh_mono_asr_test__0__speed_1 | chatgpt | 960 | 1.00 | 24.10 | 25.96 | 0.875 | 19.57 | 63.52 | 40.20 |
| en-zh_mono_asr_test__0__speed_1 | gemini | 960 | 1.00 | 17.30 | 18.90 | 0.914 | 5.62 | 185.97 | 114.00 |
| en-zh_mono_asr_test__0__speed_1 | seed_ast | 960 | 1.00 | 21.48 | 21.85 | 0.836 | -313.22 | 1.75 | 314.53 |
| en-zh_mono_asr_test__0__speed_1 | chatgpt | 1920 | 1.00 | 19.56 | 20.58 | 0.844 | 21.27 | 46.20 | 23.53 |
| en-zh_mono_asr_test__0__speed_1 | gemini | 1920 | 1.00 | 14.03 | 18.20 | 0.863 | 132.37 | 160.32 | 16.25 |
| en-zh_mono_asr_test__0__speed_1 | seed_ast | 1920 | 1.00 | 20.81 | 21.53 | 0.812 | -287.35 | 1.58 | 288.37 |
| en-zh_mono_asr_test__0__speed_1.5 | chatgpt | 960 | 1.50 | 19.74 | 20.72 | 0.807 | 27.70 | 94.63 | 65.53 |
| en-zh_mono_asr_test__0__speed_1.5 | gemini | 960 | 1.50 | 20.42 | 21.35 | 0.865 | 5.15 | 75.30 | 69.84 |
| en-zh_mono_asr_test__0__speed_1.5 | seed_ast | 960 | 1.50 | 21.13 | 21.65 | 0.805 | -221.05 | 24.40 | 241.92 |
| en-zh_mono_asr_test__0__speed_1.5 | chatgpt | 1920 | 1.50 | 16.96 | 18.31 | 0.891 | 22.70 | 40.04 | 16.09 |
| en-zh_mono_asr_test__0__speed_1.5 | gemini | 1920 | 1.50 | 20.38 | 21.44 | 0.867 | 94.40 | 109.81 | 15.09 |
| en-zh_mono_asr_test__0__speed_1.5 | seed_ast | 1920 | 1.50 | 21.30 | 21.46 | 0.818 | -202.16 | 2.13 | 203.37 |

Seed AST rows use ASR over the generated target speech with
`gpt-4o-mini-transcribe`; the AST backend translation subtitle is not used for
BLEU/chrF/CER.

## KIT Lecture Translator Status

KIT is still exploratory and should not be ranked against the full
OpenAI/Gemini/Seed rows yet. The 60s dashboard below is useful for validating
the web-event text extractor and the Chinese BLEU tokenizer issue, but it was
not run after selecting the best KIT product configuration.

The 2026-07-06 full-source KIT attempt used a default online low-latency setup
with `language=en`, `mtLanguage=zh`, `audioLanguage=zh`,
`ttsQualityMode=low_latency`, `smartChaptering=online_dynamic`, and private
availability. It was interrupted after 330.0 seconds of the 1072.63-second
FLORAS source and paused on the server. The local JSON capture is debug-only:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_full_enzh/kit_live_enzh_full_realtime_run.json
```

Before a formal full KIT comparison, sweep or inspect the relevant KIT settings
first: TTS quality/latency mode, presentation/profile selection,
postproduction, shortening, smart chaptering, pause/mute handling, and whether
target speech audio can be retrieved and ASR-scored like the Seed/GPT/Gemini
rows.

## FLORAS 60s KIT Smoke Compare

Dashboard:

```text
artifacts/compare_gpt_gemini_seed_kit_enzh_60s/index.html
artifacts/compare_gpt_gemini_seed_kit_enzh_60s_speed15/index.html
```

These rows use the same first 60s FLORAS source clip and the same 60s target
reference. The script verifies each 60s eval directory has the same
`sentence_coverage.jsonl` reference before scoring. BLEU is recomputed with
sacreBLEU `tokenize=zh`; hypothesis and reference punctuation is preserved in
`compare_metrics.jsonl` and the HTML detail panes. The old/default tokenizer
BLEU diagnostic remains in JSON, but is not shown in the dashboard.

| speed | model | chunk | variant | scope | BLEU zh | chrF | CER |
| ---: | --- | ---: | --- | --- | ---: | ---: | ---: |
| 1 | chatgpt | 0.96s | 60s-smoke-target-asr | exact_60s_source_clip | 26.24 | 25.38 | 0.704 |
| 1 | chatgpt | 1.92s | 60s-smoke-target-asr | exact_60s_source_clip | 21.21 | 21.60 | 0.729 |
| 1 | gemini | 0.96s | 60s-smoke-target-asr | exact_60s_source_clip | 26.24 | 29.01 | 0.729 |
| 1 | gemini | 1.92s | 60s-smoke-target-asr | exact_60s_source_clip | 25.34 | 27.74 | 0.696 |
| 1 | kit | 1.92s | mixed-high-quality-target-asr | exact_60s_source_clip | 25.94 | 22.72 | 0.717 |
| 1 | kit | 1.92s | online-low-latency-target-asr | exact_60s_source_clip | 24.94 | 22.60 | 0.717 |
| 1 | kit | 1.92s | online-high-quality-target-asr | exact_60s_source_clip | 23.97 | 21.17 | 0.721 |
| 1 | kit | 1.92s | online-high-quality-enonly-target-asr | exact_60s_source_clip | 20.47 | 19.83 | 0.767 |
| 1 | kit | n/a | debug-web-tts-text | debug_text_only_exact_60s_source_clip | 20.94 | 19.12 | 0.738 |
| 1 | kit | n/a | debug-web-tts-text | debug_text_only_exact_60s_source_clip | 22.99 | 21.94 | 0.767 |

Proxy rows are separated in the HTML dashboard and should not be ranked against
the exact 60s measurements:

| speed | model | chunk | variant | scope | BLEU zh | chrF | CER |
| ---: | --- | ---: | --- | --- | ---: | ---: | ---: |
| 1 | seed | 0.96s | full-run-prefix-proxy | proxy_prefix_from_full_1072s_seed_run | 28.58 | 23.99 | 0.625 |
| 1 | seed | 1.92s | full-run-prefix-proxy | proxy_prefix_from_full_1072s_seed_run | 25.98 | 24.51 | 0.596 |

KIT `*_target_asr` rows use retrieved target speech scored through
`gpt-4o-mini-transcribe`; KIT `*_tts_text` rows are debug-only web-event text.
Seed rows are not exact 60s reruns; they are prefix proxies cut from full-run
generated-target ASR using the 60s reference CJK unit count, with punctuation
kept.

## FLORAS 60s Combined Speed Compare

Dashboard:

```text
artifacts/compare_gpt_gemini_seed_kit_enzh_60s_speed15/index.html
```

This dashboard now contains both speed=1.0 rows from the 60s smoke table above
and speed=1.5 rows, grouped by speed in the HTML table. For speed=1.5, the
source content is the same first 60s EN->ZH clip, but the streamed source
speech is sped to about 40.03s.
GPT/Gemini/Seed speed=1.5 rows are not the old 60s smoke rows: they use
existing full-run generated target wavs cropped to their first 60s and
re-transcribed. KIT was run on the 60s source clip with `format=mixed`,
`ttsQualityMode=high_quality`, 1.92s input chunks, and target speech ASR.

The table below lists only the speed=1.5 subset; the speed=1.0 rows are the
60s smoke rows shown above and are included in the combined HTML dashboard.

| speed | model | chunk | variant | scope | BLEU zh | chrF | CER | source stream s | target s | dur lag s |
| ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 | chatgpt | 0.96s | full-run-target-wav-first60-asr | full_run_target_wav_first60s | 18.25 | 22.99 | 1.054 | 40.03 | 60.00 |  |
| 1.5 | chatgpt | 1.92s | full-run-target-wav-first60-asr | full_run_target_wav_first60s | 18.74 | 25.43 | 1.071 | 40.03 | 60.00 |  |
| 1.5 | gemini | 0.96s | full-run-target-wav-first60-asr | full_run_target_wav_first60s | 19.98 | 26.59 | 1.083 | 40.03 | 60.00 |  |
| 1.5 | gemini | 1.92s | full-run-target-wav-first60-asr | full_run_target_wav_first60s | 20.05 | 18.62 | 1.025 | 40.03 | 60.00 |  |
| 1.5 | seed | 0.96s | full-run-target-wav-first60-asr | full_run_target_wav_first60s | 15.76 | 23.29 | 1.642 | 40.03 | 60.00 |  |
| 1.5 | seed | 1.92s | full-run-target-wav-first60-asr | full_run_target_wav_first60s | 15.36 | 25.36 | 1.700 | 40.03 | 60.00 |  |
| 1.5 | kit | 1.92s | mixed-high-quality-target-asr | exact_60s_source_clip | 23.26 | 21.49 | 0.717 | 40.03 | 69.58 | 29.55 |

Because full-run target-wav first-60s crops can include target content beyond
the first-source-60s reference, CER can exceed 1.0 and duration lag is not
reported for GPT/Gemini/Seed crop rows. KIT remains an exact 60s source smoke
row; do not treat this table as a fully aligned formal ranking. The Seed crop
rows are especially sensitive to this windowing artifact, so their lower
BLEU/higher CER here should not be read as a direct Seed translation-quality
drop.
