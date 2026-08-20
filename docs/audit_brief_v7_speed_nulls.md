# Audit brief: null-alignment rate doubling at 1.5× source speed (v7 vs v6)

You are auditing a TTS training/eval pipeline for bugs. This brief carries the
question and numbers; **fetch the actual code from the pinned links below**
(public repo, commit `ff70b3f`) — browse any other file in the repo if you
need more context: https://github.com/luojiaxuan/S2S_omni/tree/ff70b3f

## System and observation

Cascade: InfiniSST (en→zh streaming S2T; per-speed turn streams are FIXED
files, identical inputs for every TTS model) → MOSS-TTS-Realtime finetune
(turn-by-turn TTS, constant 10-turn sliding context). Scoring: self-hosted
Qwen3-ASR-1.7B → SEGALE sentence alignment → BLEU (nulls kept as empty
hypotheses) + XCOMET-XL reference-based (nulls fixed 0.0). "null" = reference
sentence with no aligned hypothesis span (≈ under-translation).

Each cell = ONE generation (temperature 0.8, unseeded) + ONE scoring
(deterministic given audio). BLEU / XCOMET-ref / null-rate:

| model | 1× | 1.25× | 1.5× |
|---|---|---|---|
| v6 | 30.32 / 0.586 / 8.1% | 30.28 / 0.619 / 6.7% | 31.21 / 0.609 / 5.2% (15/291) |
| v7 | **31.03 / 0.652 / 2.0%** | 30.12 / 0.604 / 9.3% | 30.73 / 0.589 / **11.0% (35/317)** |

v7 = v6's training set (36,529 rows) + 12,518 rows derived from a new
trajectory-aligned corpus (6,385 passages, 157,887 turns, median turn 4–5 zh
chars — far shorter turns than v6 data, 0.96 s chunk granularity) with
mid-passage-start copies. Same base model, trainer, hyperparameters, global
batch 15, 1 epoch. Puzzle: huge null improvement at 1× (8.1→2.0%, paired
XCOMET +0.0505 t=+3.87 n=468) but doubling at 1.5× (5.2→11.0%).

## Generation-side diagnostics already run (looks healthy)

Identical turn inputs; per-turn stats of generated audio:

| run | turns | zero-frame | total audio | sec/char median |
|---|---|---|---|---|
| v6 @1.5× | 1183 | 1 | 3366 s | 0.216 |
| v7 @1.5× | 1183 | 0 | 3256 s | 0.215 |

No swallowing, same speaking rate, −3% duration. A regenerate-and-rescore
variance experiment (both models, 1.5×) is running; gen-side variance has
never been measured. Known instrument caveats: Qwen3-ASR transcribes ~22–25%
less text than gpt-4o-mini-transcribe and degrades on fast/dense speech;
under a previous GPT-ASR op, null counts varied 4/15/18 on identical audio.

## Hypotheses to weigh (find alternatives / bugs)

- H1 noise: one-sample cells; 15/291 vs 35/317 may be within gen-side variance.
- H2 distribution shift: new rows are 1×-style ultra-short turns; 1.5× streams
  are longer/denser turns outside the added coverage; the short-turn bias may
  shift prosodic boundaries at 1.5× in ways that degrade SEGALE alignment
  granularity without changing audio totals.
- H3 a real bug below.

## File index (pinned @ ff70b3f; RAW = https://raw.githubusercontent.com/luojiaxuan/S2S_omni/ff70b3f/)

v7-delta pipeline (審計重点):
- https://raw.githubusercontent.com/luojiaxuan/S2S_omni/ff70b3f/scripts/build_moss_rows_from_trajectory.py — new-corpus TSV → row requests (punct-merge, sentence-final grouping)
- https://raw.githubusercontent.com/luojiaxuan/S2S_omni/ff70b3f/scripts/synth_moss_rows_batched.py — batched in-process whole-passage synthesis of training targets (EOS/invalid/budget truncation)
- https://raw.githubusercontent.com/luojiaxuan/S2S_omni/ff70b3f/scripts/align_slice_moss_v2.py — wav2vec forced alignment + turn slicing (shared with v6 data)
- https://raw.githubusercontent.com/luojiaxuan/S2S_omni/ff70b3f/scripts/build_moss_v6_dataset.py — mid-passage-start copies + merge (seed 23)
- https://raw.githubusercontent.com/luojiaxuan/S2S_omni/ff70b3f/third_party/moss_tts/moss_tts_s2s_omni.patch — trainer patch incl. LR-scheduler fix (v6 trained 3 procs, v7 5 procs, both patched, both gbs 15)

Inference & scoring (shared by every cell):
- https://raw.githubusercontent.com/luojiaxuan/S2S_omni/ff70b3f/scripts/moss_multiturn_infer.py — turn TTS; eval uses `--sliding-window 11 --soft-reset-keep 0`
- https://raw.githubusercontent.com/luojiaxuan/S2S_omni/ff70b3f/scripts/acl_cascade_eval/score_chain_refbased.sh — full scoring chain
- https://raw.githubusercontent.com/luojiaxuan/S2S_omni/ff70b3f/scripts/build_acl6060_segale_inputs.py — SEGALE input construction
- https://raw.githubusercontent.com/luojiaxuan/S2S_omni/ff70b3f/scripts/run_acl6060_segale_alignment.py — LaBSE+vecalign alignment
- https://raw.githubusercontent.com/luojiaxuan/S2S_omni/ff70b3f/scripts/build_acl6060_xcomet_input.py — BLEU + null typing (nulls kept; see corpus_bleu and null_alignment_type)
- https://raw.githubusercontent.com/luojiaxuan/S2S_omni/ff70b3f/scripts/run_acl6060_xcomet_xl.py — XCOMET-XL, fixed_null penalty path

Context (optional): https://raw.githubusercontent.com/luojiaxuan/S2S_omni/ff70b3f/docs/experiment_ledger_moss_tts_cascade_20260808.md — full experiment ledger (sections 4.-19/4.-20 cover v7/v7b), https://raw.githubusercontent.com/luojiaxuan/S2S_omni/ff70b3f/docs/audit_bundle_v7_speed_nulls.md — a fully-inlined variant of this brief.

## Questions

1. Any bug in the v7-delta pipeline that would asymmetrically raise nulls at
   1.5× while improving them at 1×?
2. Does the generation diagnostic miss a failure mode (e.g., prosodic boundary
   drift harming sentence alignment without changing totals)?
3. Beyond regenerate-and-rescore, what would you measure next?
4. Anything suspicious about the new rows' training targets coming from an
   in-process synthesis engine while v6 rows' targets came from a serving
   engine (same model/sampling defaults)?

Answer with: (a) concrete bug findings with file/line, (b) verdict across
H1–H3 with reasoning, (c) next measurements ranked by information per hour.
