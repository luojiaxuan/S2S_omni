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
`compare_metrics.jsonl` and the HTML detail panes. The `BLEU default` column
shows the old/default tokenizer path that produced BLEU 0.0.

| label | backend | chunk_ms | scope | BLEU default | BLEU zh | chrF | CER |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| chatgpt_default_60s_asr | chatgpt | 960 | exact 60s, target-speech ASR | 0.00 | 26.24 | 25.38 | 0.704 |
| gemini_default_60s_asr | gemini | 960 | exact 60s, target-speech ASR | 0.00 | 26.24 | 29.01 | 0.729 |
| chatgpt_chunk1920_60s_asr | chatgpt | 1920 | exact 60s, target-speech ASR | 0.00 | 21.21 | 21.60 | 0.729 |
| gemini_chunk1920_60s_asr | gemini | 1920 | exact 60s, target-speech ASR | 0.00 | 25.34 | 27.74 | 0.696 |
| kit_mixed_high_quality_target_asr | kit_lecture_translator | 1920 | exact 60s, target-speech ASR | 0.00 | 25.94 | 22.72 | 0.717 |
| kit_online_low_latency_target_asr | kit_lecture_translator | 1920 | exact 60s, target-speech ASR | 0.00 | 24.94 | 22.60 | 0.717 |
| kit_online_high_quality_target_asr | kit_lecture_translator | 1920 | exact 60s, target-speech ASR | 0.00 | 23.97 | 21.17 | 0.721 |
| kit_online_high_quality_enonly_target_asr | kit_lecture_translator | 1920 | exact 60s, target-speech ASR | 0.00 | 20.47 | 19.83 | 0.767 |
| kit_fast_60s_tts_text | kit_lecture_translator | n/a | exact 60s, text-only accelerated upload | 0.00 | 20.94 | 19.12 | 0.738 |
| kit_realtime_60s_tts_text | kit_lecture_translator | n/a | exact 60s, text-only realtime upload | 0.00 | 22.99 | 21.94 | 0.767 |

Proxy rows are separated in the HTML dashboard and should not be ranked against
the exact 60s measurements:

| label | backend | chunk_ms | scope | BLEU default | BLEU zh | chrF | CER |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| seed_ast_chunk960_prefix_proxy | seed_ast | 960 | proxy prefix from full 1072s target-speech ASR | 0.00 | 28.58 | 23.99 | 0.625 |
| seed_ast_chunk1920_prefix_proxy | seed_ast | 1920 | proxy prefix from full 1072s target-speech ASR | 0.00 | 25.98 | 24.51 | 0.596 |

KIT `*_target_asr` rows use retrieved target speech scored through
`gpt-4o-mini-transcribe`; KIT `*_tts_text` rows are debug-only web-event text.
Seed rows are not exact 60s reruns; they are prefix proxies cut from full-run
generated-target ASR using the 60s reference CJK unit count, with punctuation
kept.

## FLORAS 60s Speed=1.5 KIT Smoke Compare

Dashboard:

```text
artifacts/compare_gpt_gemini_seed_kit_enzh_60s_speed15/index.html
```

The source content is the same first 60s EN->ZH clip, but the streamed source
speech is sped to about 40.03s. KIT was run with `format=mixed`,
`ttsQualityMode=high_quality`, 1.92s input chunks, and target speech ASR.

| label | backend | chunk_ms | BLEU zh | chrF | CER | source stream s | target s | dur lag s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chatgpt_default_60s_asr | chatgpt | 960 | 28.54 | 26.47 | 0.671 | 40.03 | 70.20 | 30.17 |
| gemini_default_60s_asr | gemini | 960 | 24.09 | 23.92 | 0.654 | 40.03 | 40.25 | 0.22 |
| chatgpt_chunk1920_60s_asr | chatgpt | 1920 | 24.90 | 24.09 | 0.675 | 40.03 | 69.80 | 29.77 |
| gemini_chunk1920_60s_asr | gemini | 1920 | 27.71 | 28.13 | 0.654 | 40.03 | 41.00 | 0.97 |
| kit_mixed_high_quality_speed1p5_target_asr | kit_lecture_translator | 1920 | 23.26 | 21.49 | 0.717 | 40.03 | 69.58 | 29.55 |

On this one clip, speeding source speech to 1.5x reduced KIT mixed/high-quality
BLEU from 25.94 to 23.26 and removed its speed=1.0 duration advantage: target
audio grew from 52.08s to 69.58s while the source stream was only 40.03s.
