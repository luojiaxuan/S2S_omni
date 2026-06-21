# Evaluation Plan

This project should report whether semantic compression improves streaming S2S
under source-rate stress without hiding failures behind shorter audio.

## Literature Anchors

- Hibiki reports S2S quality, speaker fidelity, naturalness, and latency. Its
  public paper uses ASR-BLEU, End Offset, LAAL, speaker similarity, and human MOS.
- IWSLT-style simultaneous translation evaluation traditionally reports text
  quality plus latency metrics such as AP, AL, DAL/LAAL.
- Recent latency papers warn that means alone hide tail failures, so summaries
  should include p50/p90/p95/max and violation rates.
- Long-form SimulS2ST evaluation recovers target transcripts/timestamps, aligns
  target and source sentence groups, then computes quality and latency at the
  alignment-group level. This matters once we evaluate audio outputs.
- Human interpreting literature treats compression, simplification, uncertainty
  handling, clarity, and ease of listening as legitimate interpreter strategies.
  Reference similarity alone will penalize some good compressed interpretations.

## Always-On Metrics

These are cheap enough to run before every SFT/RL iteration:

- `target_unit_reference_ratio`: candidate length divided by reference length,
  using CJK chars for zh/ja/ko and tokens otherwise.
- `target_budget_ratio`: candidate length divided by the assigned hard budget.
- `target_budget_violation`: true when `target_budget_ratio > 1`.
- `must_keep_term_recall`: exact target-language term recall when terms exist.
- `number_recall`: exact number preservation across source/reference.
- `bag_f1_vs_reference`: rough lexical overlap sanity check, not a semantic
  success criterion.
- `estimated_target_units_per_minute` and `estimated_target_wpm`: listenability
  pressure proxies until audio is generated.
- `end_lag_s` and `lag_penalty`: segment-level timing pressure when timestamps
  are available.
- p50/p90/p95/max summaries and violation rates, not only means.

## Split Discipline

Formal eval must be held out by original GigaSpeech `base_id`, not by stressed
variant id. All speed variants such as `AUD...__speed_1.7` and
`AUD...__speed_2` inherit the same split as `AUD...`.

- Build data with `scripts/build_gigaspeech_sft.py`, which writes
  `splits/train`, `splits/dev`, and `splits/test`.
- Root-level `compression_teacher_requests.jsonl` and `faithful_sft.jsonl` are
  train aliases kept for old commands; they are not eval files.
- Use `splits/dev/compression_teacher_requests.jsonl` for tuning and quick
  held-out checks.
- Use `splits/test/compression_teacher_requests.jsonl` only for reportable
  numbers.
- Run `scripts/verify_split_integrity.py` before reporting any held-out result
  and store the JSON summary next to the eval output.

## Optional Classic Metrics

These should be enabled depending on available artifacts:

- Text candidate vs reference: sacreBLEU, chrF, COMET/xCOMET/MetricX.
- Audio candidate with ASR transcript: ASR-BLEU/chrF/COMET against reference.
- Streaming trace: AP, AL, DAL/LAAL/FLAL, End Offset, and tail distributions.
- Audio quality: MOS-style human eval or automatic naturalness proxies.
- Voice preservation: speaker embedding cosine similarity when source speaker
  preservation is an objective.

## LLM-as-Judge

LLM judge is the main metric for the novelty claim: preserving core meaning while
compressing. It should judge:

- semantic core preservation
- unsafe critical omissions
- hallucination
- term, number, negation, modality, causality, and contrast preservation
- compression appropriateness
- listenability/ease of processing
- reference alignment while allowing paraphrase and safe omission
- whether a candidate passes the SFT target gate

The judge prompt must explicitly say that shorter wording and safe omission are
allowed; otherwise the judge collapses back to reference-literal scoring.

## SFT Gate

Before using a generated teacher label as supervised data, keep it only when:

- hard budget is satisfied or only mildly exceeded under a documented tolerance
- exact number/term checks pass when extractable
- LLM judge has no severe critical omission or hallucination
- LLM judge marks `pass_sft_gate: true`
- compression ratio is meaningfully below the faithful reference for stressed
  speed conditions

This gives us a defensible comparison:

- same reference/core-meaning alignment, better listenability than speed-up-only
  baselines; or
- same listenability/latency budget, better semantic preservation than omission
  or truncation baselines.
