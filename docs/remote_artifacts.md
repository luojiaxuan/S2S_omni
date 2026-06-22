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
The assembled 25k records contain 14,031 pass-through examples and 10,969
compression examples. The strict final audit found 0 target-character-budget
violations, 0 estimated-duration-budget violations, and 0 style-guard
violations. The speed-factor distribution is 6,878 at `1.0x`, 6,770 at
`1.35x`, 6,146 at `1.7x`, and 5,206 at `2.0x`.

`tts_requests_25k.jsonl` is a TTS sidecar for the thinker outputs. The default
backend is `qwen3_tts` with
`Qwen/Qwen3-TTS-12Hz-1.7B-Base`; it records source speech spans, target text,
target-unit counts, duration budgets, and estimated default-speech durations.

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

Temporary split rebuild location:

```text
/home/jiaxuanluo/S2S_omni/work/gigaspeech_pilot_split
```
