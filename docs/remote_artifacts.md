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
/home/sglang-omni/S2S_omni
```

Current split-aware pilot data:

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

Original GigaSpeech S2TT TSV:

```text
/mnt/taurus/data/siqiouyang/datasets/gigaspeech/train_xl_case_ft-qwen2.5-32b-instruct_marked_mfa_punc_asr.tsv
```

MFA resources:

```text
/mnt/taurus/data/siqiouyang/datasets/gigaspeech/textgrids
/mnt/gemini/data1/jiaxuanluo/gigaspeech_mfa_index/gigaspeech_mfa_index.sqlite
```

Temporary split rebuild location:

```text
/home/jiaxuanluo/S2S_omni/work/gigaspeech_pilot_split
```
