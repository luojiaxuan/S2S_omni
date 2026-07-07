## Full 1072s Sample

| run | backend | chunk_ms | speed | xCOMET-QE | MetricX-QE | MetricX err | BLEU | chrF | CER | duration_lag_s | wall_delay_s | max_backlog_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| en-zh_mono_asr_test__0__speed_1 | chatgpt | 960 | 1.00 | 0.0416 | 9.977 | 15.023 | 24.10 | 25.96 | 0.875 | 19.57 | 63.52 | 40.20 |
| en-zh_mono_asr_test__0__speed_1 | chatgpt | 1920 | 1.00 | 0.0362 | 7.761 | 17.239 | 19.56 | 20.58 | 0.844 | 21.27 | 46.20 | 23.53 |
| en-zh_mono_asr_test__0__speed_1 | gemini | 960 | 1.00 | 0.0318 | 8.686 | 16.314 | 17.30 | 18.90 | 0.914 | 5.62 | 185.97 | 114.00 |
| en-zh_mono_asr_test__0__speed_1 | gemini | 1920 | 1.00 | 0.0218 | 8.320 | 16.680 | 14.03 | 18.20 | 0.863 | 132.37 | 160.32 | 16.25 |
| en-zh_mono_asr_test__0__speed_1 | seed | 960 | 1.00 | 0.0398 | 9.943 | 15.057 | 21.48 | 21.85 | 0.836 | -313.22 | 1.75 | 314.53 |
| en-zh_mono_asr_test__0__speed_1 | seed | 1920 | 1.00 | 0.0501 | 8.362 | 16.638 | 20.81 | 21.53 | 0.812 | -287.35 | 1.58 | 288.37 |
| en-zh_mono_asr_test__0__speed_1 | kit | 960 | 1.00 | 0.0397 | 9.101 | 15.899 | 18.32 | 19.30 | 0.824 | -113.46 | 191.25 | 259.38 |
| en-zh_mono_asr_test__0__speed_1 | kit | 1920 | 1.00 | 0.0284 | 8.277 | 16.723 | 18.37 | 19.12 | 0.827 | -120.18 | 132.20 | 239.36 |
| en-zh_mono_asr_test__0__speed_1.5 | chatgpt | 960 | 1.50 | 0.0428 | 8.125 | 16.875 | 19.74 | 20.72 | 0.807 | 27.70 | 94.63 | 65.53 |
| en-zh_mono_asr_test__0__speed_1.5 | chatgpt | 1920 | 1.50 | 0.0367 | 7.744 | 17.256 | 16.96 | 18.31 | 0.891 | 22.70 | 40.04 | 16.09 |
| en-zh_mono_asr_test__0__speed_1.5 | gemini | 960 | 1.50 | 0.0342 | 8.641 | 16.359 | 20.42 | 21.35 | 0.865 | 5.15 | 75.30 | 69.84 |
| en-zh_mono_asr_test__0__speed_1.5 | gemini | 1920 | 1.50 | 0.0412 | 9.099 | 15.901 | 20.38 | 21.44 | 0.867 | 94.40 | 109.81 | 15.09 |
| en-zh_mono_asr_test__0__speed_1.5 | seed | 960 | 1.50 | 0.0431 | 9.152 | 15.848 | 21.13 | 21.65 | 0.805 | -221.05 | 24.40 | 241.92 |
| en-zh_mono_asr_test__0__speed_1.5 | seed | 1920 | 1.50 | 0.0509 | 9.432 | 15.568 | 21.30 | 21.46 | 0.818 | -202.16 | 2.13 | 203.37 |
| en-zh_mono_asr_test__0__speed_1.5 | kit | 960 | 1.50 | 0.0781 | 10.323 | 14.677 | 18.92 | 19.82 | 0.830 | 123.75 | 197.61 | 71.62 |
| en-zh_mono_asr_test__0__speed_1.5 | kit | 1920 | 1.50 | 0.0389 | 8.075 | 16.925 | 18.90 | 19.24 | 0.843 | -45.55 | 166.02 | 201.45 |

Seed AST rows use ASR over the generated target speech with
`gpt-4o-mini-transcribe`; the AST backend translation subtitle is not used for
BLEU/chrF/CER.

`xCOMET-QE` and `MetricX-QE` are reference-free source+hypothesis scores over
proportional text chunks. MetricX's native output is a lower-is-better error on
a 0-25 scale; `MetricX-QE` is reported as `25 - MetricX err` so higher is
better in the dashboards. The proportional chunks are an approximate
document-level workaround for model context limits, not sentence- or
time-aligned segments; interpret QE cautiously for rows with large backlog,
truncation, or strong compression. The current xCOMET-lite values are
diagnostic/uncalibrated rather than calibrated 0-1 xCOMET scores, because
segment values include negatives and aggregates are near zero; do not use them
as absolute quality scores until the xCOMET inference path is fixed.

The KIT rows in the full table are corrected bilingual no-post runs using
repeated `language=zh&language=en`, `mtLanguage=zh`, `audioLanguage=zh`,
`format=mixed`, `ttsQualityMode=high_quality`, and target-speech ASR. QE has
been rerun for all 16 rows in the full dashboard.

## KIT Lecture Translator Status

KIT is still exploratory. The current full-source dashboard now uses the
corrected bilingual no-post KIT full run and includes local audio/detail links:

```text
artifacts/compare_gpt_gemini_seed_kit_enzh_full/index.html
```

The active full KIT captures used `language=zh&language=en`, `mtLanguage=zh`,
`audioLanguage=zh`, `format=mixed`, `ttsQualityMode=high_quality`, private
availability, no postproduction parameter, and 0.96s/1.92s input chunks. The
hypothesis was derived from retrieved target speech transcribed by
`gpt-4o-mini-transcribe`; KIT displayed text was not used.

At 0.96s chunks, KIT scored BLEU 18.32 / chrF 19.30 / CER 0.824 with xCOMET
0.0397 / MetricX-QE 9.101 at speed=1.0, and BLEU 18.92 / chrF 19.82 / CER
0.830 with xCOMET 0.0781 / MetricX-QE 10.323 at speed=1.5. At 1.92s chunks,
KIT scored BLEU 18.37 / chrF 19.12 / CER 0.827 with xCOMET 0.0284 /
MetricX-QE 8.277 at speed=1.0, and BLEU 18.90 / chrF 19.24 / CER 0.843 with
xCOMET 0.0389 / MetricX-QE 8.075 at speed=1.5. The earlier 60s smoke advantage
still did not carry over to the full wav. The 0.96s rows are higher on the
current diagnostic QE columns, especially at speed=1.5, but this is a
single-sample result and BLEU/CER do not show a clear quality win.

The older only-en full rows scored BLEU 18.29 at speed=1.0 and BLEU 17.46 at
speed=1.5; keep them only as local historical diagnostics.

Follow-up 60s configuration checks confirmed KIT needs repeated `language`
query parameters for source language coverage. `scripts/create_kit_session.py`
now accepts repeated or comma-separated `--language` values and sends them with
`urlencode(..., doseq=True)`. On the same first-60s FLORAS clip, no-post
`format=mixed`, `ttsQualityMode=high_quality`, `audioLanguage=zh` smokes gave:

```text
language=en&language=zh: BLEU 21.74, chrF 20.69, CER 0.758, target 69.95s, 13 TTS chunks
language=zh&language=en: BLEU 20.96, chrF 20.55, CER 0.775, target 72.63s, 14 TTS chunks
```

The saved `profile_1` behind the current shorten link expands to
`language=zh,en`, `mtLanguage=zh,en`, `audioLanguage=zh`, `format=mixed`,
`ttsQualityMode=high_quality`, and `postproduction=50`. That is the product
profile, but it is not the no-compression S2S main setting; its 60s target
speech was 247.20s and scored BLEU 7.12 / chrF 15.16 / CER 3.517 against the
60s reference.

The 2026-07-06 full-source KIT attempt used a default online low-latency setup
with `language=en`, `mtLanguage=zh`, `audioLanguage=zh`,
`ttsQualityMode=low_latency`, `smartChaptering=online_dynamic`, and private
availability. It was interrupted after 330.0 seconds of the 1072.63-second
FLORAS source and paused on the server. The local JSON capture is debug-only:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/floras_live_pilot_refs/kit_full_enzh/kit_live_enzh_full_realtime_run.json
```

Before treating KIT as a product-level product claim, sweep or inspect the
remaining relevant settings: presentation/profile selection, postproduction,
shortening, smart chaptering, pause/mute handling, and whether other settings
reproduce the same target-speech-ASR behavior. The next metric maintenance step
is to rerun xCOMET/MetricX QE for the replacement KIT rows.

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

Additional 2026-07-06 KIT smokes after the source-language correction are not
yet part of the tracked 60s dashboard table. The no-post bilingual
`format=mixed` rows scored BLEU 21.74 for `language=en&language=zh` and BLEU
20.96 for `language=zh&language=en`; the current saved profile with
`postproduction=50` produced an overlong 247.20s target and BLEU 7.12. These
results explain why the old full only-en rows should be discarded. The
corrected full run has now been added to the main full dashboard, and its
quality is lower than the 60s smoke suggested.

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
