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
