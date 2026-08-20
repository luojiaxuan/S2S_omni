# Audit brief: v7 + soft-reset(keep=3) — audible repetition, higher measured under-translation, no perceived coherence gain

You are auditing a streaming-TTS inference mode and its evaluation. Fetch code
from the pinned links (public repo, commit `2976ed9`); browse anything else at
https://github.com/luojiaxuan/S2S_omni/tree/2976ed9

## System

InfiniSST (en→zh S2T; fixed per-speed turn streams) → MOSS-TTS-Realtime
finetune "v7", turn-by-turn TTS. Two context modes in
`scripts/moss_multiturn_infer.py`:

- **constant sliding** (`--sliding-window 11 --soft-reset-keep 0`): context =
  last 10 turns, rebuilt every turn, KV cache never reusable;
- **soft reset** (`--soft-reset-keep 3`, lines ~498-508): window grows to 10
  (pure append, cache reusable) then shrinks to last 3; cycle of ~8 turns.

Per-turn audio is np.concatenate'd with no crossfade and no boundary silence
in both modes. Scoring: self-hosted Qwen3-ASR-1.7B → spaCy sentence split →
LaBSE+vecalign (SEGALE) vs 468 fixed reference sentences → sentence BLEU
(nulls kept) + XCOMET-XL-ref (nulls zeroed) + under-translation rate
(distinct ref sentences in under blocks / 468).

## Observations (the puzzle)

1. **Human listening**: user reports soft3 sounds no more coherent than
   constant sliding, and has *obvious repetition*. The file they heard first
   (soft3 1×, talk110) opens with a degenerate **turn 0**: input text was
   only "大家好，这是" but the model produced a 15.0s hallucinated podcast
   opener with a verbatim internal repeat ("……希望接下来的内容能让你有所收
   获希望接下来……"), right at 0:00. Constant sliding's turn 0 on the same
   talk is a 10.96s version of the same hallucination.
2. **Yet quantitatively soft3 has LESS repetition** (per-turn independent
   Qwen3-ASR over turn slices; ≥4-char turns):

   | run | intra-turn loop (≥3 dup 6-grams in one turn's ASR) | cross-turn re-read (ASR contains ≥6-char substring of context turns, not own text) |
   |---|---|---|
   | sliding 1× | 3/1479 (0.2%) | 16/1137 (1.4%) |
   | soft3 1× | 1/1479 (0.1%) | 17/1099 (1.5%, uniform across shrink cycle) |
   | sliding 1.5× | 6/1088 (0.6%) | 17/979 (1.7%) |
   | soft3 1.5× | 0/1088 (0.0%) | 10/974 (1.0%) |

   Re-reads do NOT cluster after the shrink boundary (cycle-position
   histogram flat), so the window mechanic itself looks innocent.
3. **Scores (fixed-ref op, BLEU / XCOMET-ref / under-rate)**:

   | v7 mode | 1× | 1.5× |
   |---|---|---|
   | sliding | 31.03 / 0.652 / 1.3% (6/468) | 30.73 / 0.589 / 7.5% (35/468) |
   | soft3 | 28.51 / 0.604 / 6.8% (32/468) | 30.47 / 0.648 / 3.8% (18/468) |

   Perfect mirror crossover. Null-flip test (search each missing ref's
   content in per-turn ASR): soft3 1× → 21/32 present in audio ("artifact"),
   8 marginal, only 3 true misses; soft3 1.5× → 10/8/**0**. So soft3's worse
   1× under-rate is mostly the instrument (coherent prosody → fewer ASR
   sentence breaks → worse vecalign granularity), a mechanism already
   documented in the ledger for v6.
4. Historical context: under a previous GPT-ASR + paired-XCOMET op, v6+soft3
   BEAT v6+sliding at 1× (paired t=+9.7). Under the current Qwen op it loses
   at 1× for both v6 and v7. Deployment default is soft3; canonical eval
   pins sliding.

## Files (RAW = https://raw.githubusercontent.com/luojiaxuan/S2S_omni/2976ed9/)

- RAW/scripts/moss_multiturn_infer.py — both modes; window logic ~498-508;
  loop-detect + higher-temperature regen path ~440-490; first-turn prompt
  assembly (search `first_prompt` / `make_ensemble` / `im_start`); sanitize;
  seed default 42 (~47).
- RAW/scripts/acl_cascade_eval/run_eval_queue.sh — exact eval invocations.
- RAW/scripts/build_acl6060_segale_inputs.py, RAW/scripts/run_acl6060_segale_alignment.py,
  RAW/scripts/build_acl6060_xcomet_input.py — scoring chain (fixed-ref under
  metric in the last one).
- Context: RAW/docs/experiment_ledger_moss_tts_cascade_20260808.md
  (sections 4.-21 … 4.-21c), RAW/docs/audit_chatgpt_v7_speed_nulls_20260820.md
  (your previous audit).
- Audio to reason about (public):
  https://huggingface.co/datasets/gavinlaw/s2s-omni-cascade-demo-audio —
  audio/v7soft3_1x_talk110.mp3 vs audio/v7_1x_talk110.mp3 etc.

## Questions

1. **Turn-0 hallucination**: mechanism? The prompt starts with a voice-prompt
   token ensemble + "<|im_start|>assistant\n" then the first tiny text turn.
   Why does the model invent long podcast-style filler with internal repeats
   on the FIRST turn only, in both modes? Propose minimal fixes (prompt-side,
   decoding-side, or post-hoc trim) and how to validate cheaply.
2. **Any real defect in the soft-reset implementation** (window update,
   prompt rebuild at shrink, loop-regen interaction, cache reuse) that could
   produce audible artifacts despite the flat cycle-position histogram?
3. **Instrument fairness**: is the sentence-level op structurally biased
   against coherent prosody? What minimal instrument change (e.g., inserting
   150-300ms silence at turn seams before ASR, or scoring per-turn ASR
   directly) would make the mode comparison fair without abandoning the
   canonical chain? Would you expect soft3's 1× gap to close?
4. **Operating-point recommendation**: given the mirror crossover, the
   null-flip evidence, and a hard product constraint of shipping ONE
   checkpoint and ideally one mode, what would you ship today, and what
   single experiment most reduces the risk of that choice being wrong?

Answer with: (a) findings with file/line, (b) direct answers to 1-4 ranked by
confidence, (c) cheapest-first measurement plan.
