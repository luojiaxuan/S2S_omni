# Remote Artifacts

This repo tracks code, configs, prompts, and docs. Large generated data, model
outputs, and checkpoints stay outside git.

## Local Workspace

Project root:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/work/S2S_omni
```

## b200

Project mirror:

```text
/data/repo/S2S_omni
```

Host-side persistent mirror:

```text
/data/sglang-omni/s2s_omni_data/repo/S2S_omni
```

Current 25k natural-policy SFT dataset, generated on 2026-06-22:

```text
/data/repo/S2S_omni/work/gigaspeech_policy_pool_30k_lazy_20260622
```

Important files:

```text
work/gigaspeech_policy_pool_30k_lazy_20260622/policy_sample_manifest.jsonl
work/gigaspeech_policy_pool_30k_lazy_20260622/sft_25k.jsonl
work/gigaspeech_policy_pool_30k_lazy_20260622/manifest_25k.jsonl
work/gigaspeech_policy_pool_30k_lazy_20260622/tts_requests_25k.jsonl
work/gigaspeech_policy_pool_30k_lazy_20260622/sft_25k_summary.json
work/gigaspeech_policy_pool_30k_lazy_20260622/sft_25k_rejected.jsonl
```

The final dataset uses the natural RTF policy over a 30k GigaSpeech policy
candidate pool, without manually fixing the pass-through/compression ratio.
The assembled 25k records contain 13,807 pass-through examples and 11,193
compression examples. The strict final audit found 0 target-character-budget
violations, 0 estimated-duration-budget violations, and 0 style-guard
violations. The speed-factor distribution is 6,786 at `1.0x`, 6,726 at
`1.35x`, 6,220 at `1.7x`, and 5,268 at `2.0x`.

`tts_requests_25k.jsonl` is a TTS sidecar for the thinker outputs. The default
backend is `qwen3_tts` with
`Qwen/Qwen3-TTS-12Hz-1.7B-Base` and SGLang-Omni example config
`examples/configs/qwen3_tts_1_7b.yaml`; it records source speech spans, target
text, target-unit counts, duration budgets, and estimated default-speech
durations.

Held-out dev/test policy eval data, generated on 2026-06-22 after excluding all
25k train `base_id`s:

```text
/data/repo/S2S_omni/work/gigaspeech_heldout_eval_20260622
```

The held-out split has 300 dev and 300 test policy samples, with no dev/test
`base_id` overlap and no train `base_id` overlap. Each split is roughly half
pass-through and half compression under the default-speech S2S RTF policy.

TTS duration spot-check output:

```text
/data/outputs/s2s_tts_duration_audit_4_20260622
```

The 4-sample check synthesized faithful reference text and compressed target
text with Qwen3-Omni talker. All 4 compressed outputs were shorter than their
faithful-reference counterparts; observed compressed/reference duration ratios
were 0.51, 0.58, 0.61, and 0.67.

Current trainability note: `sft_25k.jsonl` provides thinker/text supervision.
Talker LoRA requires audio codec supervision (`labels`/`residual_codes` style
inputs in the HF talker forward path), so it needs a separate TTS-to-code-label
pipeline before it can be trained as a genuine supervised talker adapter.

Final 25k thinker-LoRA SFT run, completed on 2026-06-22:

```text
/data/checkpoints/s2s_omni/qwen3_omni_25k_thinker_lora_20260622_full
```

The run trained for 1 epoch over 25k policy samples and finished 1,563 optimizer
steps with final reported `train_loss=10.66`. The final adapter files are
`adapter_model.safetensors` and `adapter_config.json`.

Held-out eval output for the final adapter:

```text
/data/outputs/s2s_eval_qwen3_omni_25k_thinker_lora_20260622_full
```

Local synced copy:

```text
/Users/luojiaxuan/Documents/Codex/2026-06-20/s/outputs/s2s_eval_qwen3_omni_25k_thinker_lora_20260622_full
```

Key held-out metrics:

| Split | System | chrF | BLEU | Bag-F1 | Target/ref | Budget ratio | S2S RTF | RTF violation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | base | 23.50 | 0.021 | 0.594 | 0.705 | 0.694 | 0.691 | 10.67% |
| dev | evo | 34.22 | 0.577 | 0.674 | 0.821 | 0.772 | 0.768 | 7.33% |
| test | base | 22.17 | 0.041 | 0.580 | 0.720 | 0.702 | 0.699 | 13.33% |
| test | evo | 33.65 | 0.702 | 0.673 | 0.815 | 0.767 | 0.763 | 6.67% |

Audio sample page:

```text
/data/outputs/s2s_eval_qwen3_omni_25k_thinker_lora_20260622_full/audio_samples_dev/index.html
```

The generated-audio sample audit covers 8 dev examples with base/evo paired
Qwen3-Omni audio. Evo improved text coverage on average while keeping estimated
S2S RTF near the base level, but the true generated wav durations did not
reliably shrink relative to base: base mean wav duration was 12.87s and evo mean
wav duration was 15.81s. This confirms that thinker-only LoRA plus frozen talker
does not guarantee audio duration control, even when the text policy satisfies
the estimated duration budget.

Current RTF-aware split pilot data:

```text
/home/sglang-omni/S2S_omni/work/gigaspeech_pilot_rtf
```

Important files:

```text
work/gigaspeech_pilot_rtf/splits/train/pass_through_sft.jsonl
work/gigaspeech_pilot_rtf/splits/train/compression_teacher_requests.jsonl
work/gigaspeech_pilot_rtf/splits/train/faithful_sft.jsonl
work/gigaspeech_pilot_rtf/splits/dev/rtf_decision_manifest.jsonl
work/gigaspeech_pilot_rtf/splits/test/rtf_decision_manifest.jsonl
work/gigaspeech_pilot_rtf/split_integrity_rtf_decision_b200.json
```

The compression decision is based on default-speech S2S real-time factor:
examples go to `pass_through_sft.jsonl` when the faithful target speech fits
before the next source chunk, otherwise they go to
`compression_teacher_requests.jsonl`.

Previous split-aware pilot data:

```text
/home/sglang-omni/S2S_omni/work/gigaspeech_pilot_split
```

Important files:

```text
work/gigaspeech_pilot_split/splits/train/compression_teacher_requests.jsonl
work/gigaspeech_pilot_split/splits/dev/compression_teacher_requests.jsonl
work/gigaspeech_pilot_split/splits/test/compression_teacher_requests.jsonl
work/gigaspeech_pilot_split/splits/train/faithful_sft.jsonl
work/gigaspeech_pilot_split/split_integrity_teacher_requests_b200.json
```

The root-level files under `work/gigaspeech_pilot_split/` are train aliases kept
for older commands. Formal held-out evaluation should use `splits/dev` or
`splits/test`.

Previous thinker-only LoRA run:

```text
~/S2S_omni/runs/qwen3_omni_gigaspeech_mixed_compression_sft_20260621_092701
```

## taurus

Persistent project mirror:

```text
/mnt/data2/jiaxuanluo/S2S_omni
```

Original GigaSpeech S2TT TSV:

```text
/mnt/taurus/data/siqiouyang/datasets/gigaspeech/train_xl_case_ft-qwen2.5-32b-instruct_marked_mfa_punc_asr.tsv
```

MFA resources:

```text
/mnt/taurus/data/siqiouyang/datasets/gigaspeech/textgrids
/mnt/gemini/data1/jiaxuanluo/gigaspeech_mfa_index/gigaspeech_mfa_index.sqlite
```

Full-corpus RTF distribution summary, generated on 2026-06-22:

```text
/mnt/data2/jiaxuanluo/S2S_omni/work/gigaspeech_full_rtf_summary_20260622/build_summary.json
```

The run scanned 1,365,025 TSV records, kept 1,281,846 usable rows after text and
duration filters, and evaluated speed factors `1.0,1.35,1.7,2.0`. Across all
speed variants the natural pass-through/compression split was 2,574,540 /
2,552,844, or 50.2% / 49.8%. By speed factor, compression rates were 3.1% at
`1.0x`, 37.7% at `1.35x`, 72.3% at `1.7x`, and 86.1% at `2.0x`.

The final 25k natural-policy SFT dataset is mirrored from b200 here:

```text
/mnt/data2/jiaxuanluo/S2S_omni/work/gigaspeech_policy_pool_30k_lazy_20260622
```

Mirrored final files:

```text
sft_25k.jsonl
manifest_25k.jsonl
tts_requests_25k.jsonl
sft_25k_summary.json
sft_25k_rejected.jsonl
```

Temporary split rebuild location:

```text
/home/jiaxuanluo/S2S_omni/work/gigaspeech_pilot_split
```

## aries

Recommended workspace for the current RASST soft-wav E2E SFT route:

```text
/mnt/data/jiaxuanluo/S2S_omni
```

Recommended run root:

```text
/mnt/data/jiaxuanluo/S2S_omni_runs
```

Soft-wav Python user base:

```text
/mnt/data/jiaxuanluo/python_userbase/s2s_omni_softwav
```

MFA is installed as a separate micromamba environment, because the pip package
does not include the required Kaldi/kalpy backend:

```text
/mnt/data/jiaxuanluo/micromamba/envs/mfa/bin/mfa
```

Use the larger local cache path for Qwen3-Omni weights:

```text
/mnt/data3/jiaxuanluo/.cache/huggingface
```

Plain RASST baseline zh data used by the soft-wav route:

```text
/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline.jsonl
/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline_dev.jsonl
```

These are 12,500 train rows and 355 dev rows, with ordinary assistant Chinese
chunk targets. They are not the term-map/tagged files used by the
`model_infinisst_baseline` checkpoint args. The soft-wav route generates target
speech with original `Qwen/Qwen3-Omni-30B-A3B-Instruct`, captures Omni-compatible
codes, aligns full target wav/text with MFA, and trains thinker+talker LoRA while
freezing `code2wav`.

Planned Omni-compatible wav2codec self-domain pair data:

```text
/mnt/data2/jiaxuanluo/S2S_omni/work/omni_s2s_codec_pairs_25k_YYYYMMDD
```

Planned wav2codec checkpoints:

```text
/mnt/data2/jiaxuanluo/S2S_omni/checkpoints/wav2omni_codec_25k_YYYYMMDD
```

Default execution env:

```text
/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python
```

Use `scripts/run_wav2codec_pipeline_slurm.sh` with `PARTITION=aries` or
`PARTITION=taurus`. The first version targets Omni self-domain reconstruction,
not external TTS wav generalization.
