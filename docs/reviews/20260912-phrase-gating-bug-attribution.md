# GPT 外审:turn-end loss bug 的归因与重训方案(2026-09-12 PT)

模型档位:最新旗舰「极高」(次高推理档),临时聊天。
我方送审的判断见 `projects/infinisst_moss_tts_cascade/research_log.md` 2026-09-12 条目。

## 全文(原样落盘,未删减)

Attribution "the loss bug caused the phrase arm to lose" is currently too strong; "the bug is a high-probability explanation for the phrase arm's relative degradation" is justified. The strongest evidence is the treatment interaction you already have: empty turns are 29.3% phrase vs 10.9% word-aligned = 2.69x exposure, and the missing supervision is exactly on the action "emit nothing / wait," which phrase gating requires disproportionately often.

The +5.05 BLEU external delta is supporting mechanism evidence, not a quantitative explanation of your -2.83 BLEU gap. Their 36.87 -> 41.92 shows the loss defect can be large enough to matter, but differences in data, training trajectory, gating distribution, and evaluation prevent saying "+5.05 > 2.83, therefore the bug explains our deficit."

The strongest alternative explanation is that your hand-written gating policy itself creates worse segmentation/context allocation than word alignment. >=8 readable chars and punctuation + >=4 optimize surface form rather than semantic completeness; they can hold useful short phrases too long, split long semantic units at arbitrary punctuation, alter TTS conditioning lengths, and increase latency-conditioned translation errors even with a perfectly trained silence policy.

A second strong alternative is distribution shift from the phrase corpus beyond empty-turn frequency. If phrase construction changes assistant-turn length, number of turns/sample, previous-context composition, punctuation distribution, or target-token positions, fixing empty-turn supervision may improve phrase training without eliminating those disadvantages.

A third alternative is that the patch helps generic turn-boundary learning rather than specifically silence learning. Weighting empty-turn end tokens changes the learned probability of terminating/skipping a turn; phrase gating benefits more because it encounters more empty turns, but that does not establish "zero gradient on silence" as the sole causal pathway.

| Test | What it establishes | What it does not establish |
|---|---|---|
| Old word 40.59 vs old phrase 39.80 | Old pipeline favors word by +0.79 BLEU | Cause |
| Retrain both with empty_turn_end_w=0.5 | Fair comparison under corrected objective | General superiority of phrase gating |
| Phrase improvement >> word improvement | Differential sensitivity consistent with empty-turn bug | Unique causal mechanism |
| Fixed-loss phrase > fixed-loss word | Phrase gating helps under this heuristic/configuration | LLM semantic phrases are unnecessary |
| Fixed-loss phrase <= fixed-loss word | Bug insufficient to rescue heuristic phrase gating | Phrase gating as a concept is bad |

Retraining both arms with everything else frozen is exactly the correct next experiment; do not change the gating heuristic yet. Changing 8-char gating to LLM semantic segmentation simultaneously changes the loss and the intervention, destroying the cleanest causal test you currently have.

Use the existing old-loss checkpoints plus the two corrected-loss checkpoints as a 2x2 interaction test, not merely a new A/B. Define Delta_word = M(W_fixed) - M(W_old), Delta_phrase = M(P_fixed) - M(P_old), and report I = Delta_phrase - Delta_word. A materially positive I is the key evidence that the bug disproportionately damaged phrase training.

Your most informative outcome is not simply P_fixed > W_fixed; it is I > 0 together with P_fixed > W_fixed. The first supports the bug attribution; the second supports the narrower claim that corrected phrase gating beats corrected word alignment.

Predeclare the interpretation before training. If P_fixed > W_fixed on both BLEU and XCOMET, claim "phrase gating improves quality under corrected empty-turn supervision"; if BLEU/XCOMET split, claim a metric tradeoff; if both remain worse, reject the bug as a sufficient explanation and investigate the gating policy next.

Do not claim "phrase gating helps" generically from five talks. The defensible claim is "with this training recipe, chunk size, TTS cascade, and hand-written gate, phrase gating outperforms word alignment on the held-out evaluation after correcting empty-turn supervision."

After the clean bug test, replacing the heuristic with semantic phrase boundaries is the highest-value next ablation if fixed-loss phrase still loses or only ties. Compare exactly {word-aligned, 8-char heuristic, semantic-phrase} under the same corrected loss; otherwise you cannot separate "phrase-level waiting is useful" from "this particular phrase detector is good."

One deterministic inference run per checkpoint is enough to eliminate inference-repeat requirements if greedy decoding and the entire preprocessing/audio pipeline are deterministic. Re-running the identical checkpoint on the identical five talks provides essentially no new statistical information; "ASR noise" that is deterministic conditional on the waveform is error, not Monte-Carlo variance.

Training stochasticity is the reason repeats may still be required. If initialization, data order, dropout, distributed reduction, or optimizer trajectory varies between retrains, one checkpoint per arm confounds the loss/gating effect with training-seed variance even though inference is deterministic.

Do one paired retrain per arm first; require multi-seed retraining only if the corrected A/B is close enough that seed variation could reverse the conclusion. A practical rule is: |P_fixed - W_fixed| < ~1 BLEU or contradictory BLEU/XCOMET -> run >=3 matched training seeds; a large coherent margin such as >2 BLEU plus improved XCOMET justifies proceeding to broader evaluation before burning compute on replicas.

Do not use repeated inference as the replication mechanism; use paired training seeds and/or more independent talks. Five talks are the larger external-validity problem because correlated sentences from the same talk do not create independent experimental units; report per-talk deltas and preferably a talk-clustered paired bootstrap when the final test set is larger.

Freeze these variables for the immediate rerun: training corpus construction, checkpoints/init, optimizer, schedule, data order/seed where feasible, chunk duration 1.92 s, decoding, TTS, gate definitions, evaluation set, BLEU/XCOMET scripts, and only change empty_turn_end_w: 0 -> 0.5.

Record one diagnostic in addition to BLEU/XCOMET: silence-decision accuracy conditioned on chunks whose training target is empty. The causal prediction is specific: corrected loss should substantially improve P(silence | empty-target chunk) or reduce premature emissions, with a larger change in the phrase arm; quality improvement without any change in silence behavior weakens your proposed mechanism.

Best current wording for the paper claim: "The original phrase-gated comparison was confounded by a loss implementation that provided no supervised token for empty assistant turns; because phrase-gated training contains 29.3% empty turns versus 10.9% for word-aligned training, we re-evaluate both methods under explicit empty-turn-end supervision."
