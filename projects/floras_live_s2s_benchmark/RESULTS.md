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

## FLORAS 60s KIT Smoke Compare

Dashboard:

```text
artifacts/compare_gpt_gemini_seed_kit_enzh_60s/index.html
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
| kit_fast_60s_tts_text | kit_lecture_translator | n/a | exact 60s, text-only accelerated upload | 0.00 | 20.94 | 19.12 | 0.738 |
| kit_realtime_60s_tts_text | kit_lecture_translator | n/a | exact 60s, text-only realtime upload | 0.00 | 22.99 | 21.94 | 0.767 |

Proxy rows are separated in the HTML dashboard and should not be ranked against
the exact 60s measurements:

| label | backend | chunk_ms | scope | BLEU default | BLEU zh | chrF | CER |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| seed_ast_chunk960_prefix_proxy | seed_ast | 960 | proxy prefix from full 1072s target-speech ASR | 0.00 | 28.58 | 23.99 | 0.625 |
| seed_ast_chunk1920_prefix_proxy | seed_ast | 1920 | proxy prefix from full 1072s target-speech ASR | 0.00 | 25.98 | 24.51 | 0.596 |

KIT rows use captured web-event TTS text because the target wav was not
retrieved from the product. Seed rows are not exact 60s reruns; they are prefix
proxies cut from full-run generated-target ASR using the 60s reference CJK unit
count, with punctuation kept.
