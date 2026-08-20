# ChatGPT 外部审计回执：v7@1.5× 漏译翻倍（2026-08-20）

输入是 `docs/audit_brief_v7_speed_nulls.md`（git + 文件索引方案，pinned
`ff70b3f`）；执行环境：网页 ChatGPT，推理档 Extra High，联网浏览 repo，
标注耗时 13m21s。以下为回复全文（英文原文，未改动；结论的本地核验与
采纳情况见台账 4.-21 与 6.11）。

---

I audited the pinned ff70b3f code, including the pinned SEGALE implementation and the MOSS-TTS upstream revision used by the repo. My main conclusion is: there is no obvious v7-only code path that directly activates at 1.5×, but there are real bugs in both the training-target slicer and the evaluation metric, and the latter makes the reported 5.2%→11.0% comparison not the fixed-reference null rate you think it is. The remaining speed interaction is very plausibly H2-style prosody/boundary behavior amplified by Qwen→spaCy→SEGALE.

## (a) Concrete findings

| Finding | Severity | Relevance to 1.5× |
| --- | --- | --- |
| "Null rate" denominator is model-dependent SEGALE alignment blocks, not fixed reference sentences | Confirmed bug / metric mismatch | Very high |
| Reported null combines under- and over-translation | Confirmed metric mismatch | Medium |
| Forced-align fallback redistributes all missing turns into the first missing gap | Confirmed code bug | High for v7 because turns are ultra-short |
| "Every turn keeps at least one frame" repair can still leave trailing 0-frame targets | Confirmed code bug | Medium/high if slicer bug activates |
| Eval generation is seeded 42 by default, contrary to "unseeded" | Confirmed reproducibility mismatch | High for the running variance experiment |
| Paired XCOMET t-test treats replicated block scores as 468 independent sentence observations | Statistical bug / anti-conservative SE | Medium |
| v7 synthesis claims serving defaults .8/.9/50, while pinned upstream server defaults are .8/.6/30 | Unresolved parity bug | Medium |
| Scoring shell hard-codes a _speed1 rundir and calls an external, unpinned score_generic.py | Audit/reproducibility hazard | Potentially critical; verify manifests |

### 1. The biggest issue: 15/291 vs 35/317 is not a fixed-reference null rate

build_acl6060_segale_inputs.py constructs a fixed reference row for every source sentence and records source_segments; those fixed units are invariant for a given speed/run input.

But pinned SEGALE then: sentence-splits the model-dependent ASR transcript with spaCy; vecaligns those hypothesis sentences against the fixed source sentences; permits many-to-many mappings; emits one JSON row per alignment block, with a list of src_ref_ids.

Then build_acl6060_xcomet_input.py:134-155 sets: rows = alignment blocks, segments = len(rows), null_count = over_count + under_count, null_alignment_ratio = null_count / len(rows).

So 291 and 317 are counts of SEGALE alignment blocks, not counts of reference sentences. The fact that the denominator changes from 291 to 317 on the same fixed 1.5× source set is the smoking gun.

Moreover, a null block containing three src_ref_ids counts as one null, not three missing reference sentences. And lines 138-153 combine over_translation with under_translation, even though your definition of "null" is specifically "reference sentence with no aligned hypothesis."

The corrected statistic from already-existing files is straightforward:

- numerator = number of distinct source_segment_ids belonging to under_translation blocks
- denominator = input_summary["source_segments"]

Report over-translation separately.

I would not interpret 5.2% vs 11.0% at all until you recompute this. The true fixed-unit result may still favor v6 substantially; the point is that we currently do not know.

The same model-dependent segmentation also affects BLEU, because corpus_bleu() is fed the SEGALE blocks, and affects system XCOMET because it is an arithmetic mean across those blocks. This is consistent with your own earlier ledger finding that document BLEU removes a large SEGALE penalty caused partly by continuous prosody yielding long ASR sentences and unfavorable alignment granularity.

### 2. There is a real bug in align_slice_moss_v2.py:114-130

The intended behavior is to interpolate a contiguous run of turns that contain no alignable wav2vec characters. Instead:

`gap_segs = [j for j in range(n) if starts[j] is None ...]`

collects every unaligned segment anywhere in the row, then assigns all of them proportionally into the interval between the current preceding aligned turn and the current next aligned turn, and immediately breaks.

Example shape: aligned, missing, aligned, aligned, missing, aligned — the second missing segment—far later in the passage—gets assigned into the first gap.

That can create temporally non-monotonic spans. boundaries_from_spans() then clamps backwards boundaries forward, after which frame-cut repair forces duplicate cuts apart one frame at a time.

This is shared with v6, so it is not intrinsically a 1.5× bug. But it is highly v7-amplified: the v7 corpus has huge numbers of 4–5-character turns, and the code explicitly says digits/Latin/rare Hanzi can disappear from the aligner vocabulary, leaving whole turns with no aligned chars.

There is a second bug at align_slice_moss_v2.py:211-213: the comment promises "every turn keeps at least one frame," but once frame_cuts[k-1] == total_frames, min(total_frames, previous+1) cannot advance. All remaining turns can therefore receive zero audio frames. The MOSS SFT packer still places an EOS target on an assistant turn with zero audio frames, so such examples effectively supervise an immediate stop.

That gives you a credible causal H3→H2 bridge: ultra-short v7 turns activate a slicing defect → noisy one-frame/zero-frame acoustic targets and excess EOS/boundary supervision → model changes termination/prosody behavior.

I would not assert that it caused the 1.5× result until you measure how often it fires.

### 3. Your "unseeded one-generation cell" description conflicts with the pinned inference code

moss_multiturn_infer.py:35-43 defaults --seed 42, and lines 104-106 call both torch.manual_seed() and torch.cuda.manual_seed_all(). The pinned eval launcher invokes this script without --seed, so unless another actual launcher overrides it, every run is getting seed 42.

This matters immediately for the regenerate-and-rescore experiment: rerunning the same command is not a generation-variance experiment. It may reproduce the same sampled trajectory exactly or very nearly. Use explicitly different seeds—e.g. 41/42/43/44/45—and log them into the output record.

So I downgrade H1 as currently formulated: "unseeded one draw" is not what the checked-in code does.

### 4. Paired XCOMET t=3.87, n=468 is better than the raw system mean, but its p-value is too optimistic

paired_xcomet_diff.py takes one SEGALE block score and assigns that same score to every source_segment_id contained in the block, then computes SE = stdev(diff)/sqrt(n) over the resulting source IDs. If a block covers 2–4 reference sentences, those observations share the exact same XCOMET measurement. They are not statistically independent. Sentences within the same talk are correlated as well.

So I would retain the +0.0505 mean effect as useful evidence of a real 1× improvement, but not quote t=3.87 as calibrated significance. Bootstrap by talk/document, or at minimum cluster by SEGALE block. Also note that different v6/v7 alignment groupings mean the paired score remains segmentation-dependent.

### 5. The v7 trajectory construction strongly supports H2 even without a "bug"

build_moss_rows_from_trajectory.py:66-82 drops empty trajectory increments and treats each remaining nonempty increment as a TTS turn. It then groups many of those tiny turns into synthesis requests up to roughly 300 characters. synth_moss_rows_batched.py synthesizes each whole group continuously, then concatenates group waveforms. Afterward, align_slice_moss_v2.py cuts that continuous speech back into 4–5-character assistant turns, and the training packer attaches an EOS supervision event to every resulting assistant turn.

That is a substantial training-distribution change even if every boundary were perfectly aligned: many "turn boundaries" in v7 are not acoustic sentence boundaries at all. They are arbitrary trajectory increments carved out of a continuously spoken passage.

This is almost tailor-made to produce your pattern: at 1×, where inference also contains short turns, the added data teaches the model not to swallow them; at 1.5×, turns are longer/denser and the learned boundary/EOS prior is off-distribution; the output can remain perfectly healthy in total seconds and sec/char while punctuation, pause placement, F0 reset, or phrase grouping changes.

### 6. Yes—the generation diagnostic misses exactly the failure mode you suggested

Inference synthesizes each TTS turn independently into turn_audio, then simply row_pcm.append(turn_audio) and ultimately np.concatenate(row_pcm). There is no inserted boundary silence. Therefore these two generations can have identical: number of nonzero turns, total duration, median sec/char, while differing radically in: pause duration at turn seams, whether a seam has an F0/prosodic reset, whether the previous phrase "pulls" the next turn into it, articulation around the seam, Qwen punctuation, Qwen sentence segmentation, ASR loss concentrated in ±300 ms around seams. And SEGALE explicitly sentence-splits that Qwen transcript before vecalign.

Your own ledger has already observed essentially this mechanism: continuous prosody → longer ASR sentence splits → harsher SEGALE alignment, and document BLEU narrowed the apparent gap. So yes: prosodic-boundary drift is the first generation-side failure mode I would investigate now.

### 7. The in-process-vs-serving equivalence is not established

The v7 batch synthesizer uses defaults: temperature=.8, top_p=.9, top_k=50. The batch=1 in-process script explicitly says .8/.9/50 are the "MOSS session defaults" and claims the old v2 serving requests therefore used the same values. But the MOSS-TTS upstream server at the revision pinned by this repo has: temperature=.8, top_p=.6, top_k=30.

Meanwhile, the actual v6 target-generation client sends no sampling parameters whatsoever; it posts only model/voice/input/format/ref-audio, so the outcome depends entirely on whatever defaults were present in the historical serving container. And your ledger explicitly says that old serving stack is buried in an unreproducible container layer.

Therefore: "v6 and v7 used the same sampling defaults" is currently an unsupported assumption. If archived server logs/config show .9/.50, this concern disappears. If they show .6/.30, you have a real v7 target-generation distribution change.

The batched implementation also changes exact RNG semantics because it samples lockstep batches with a shard seed, so a sample depends on batch composition. I view that as a reproducibility/parity issue rather than a likely systematic quality defect.

### 8. One scoring-path issue deserves an immediate provenance check

score_chain_refbased.sh:11 hard-codes ..._chunk192_speed1 for $RD, and then calls an external $BENCH/score_generic.py; that scorer is not in the public pinned repository. The experiment ledger itself records a previous score_generic/PREFIX drift.

I am not saying your 1.5× data were scored as 1×—the external scorer may deliberately populate this canonical directory according to PREFIX. But the public commit is insufficient to prove otherwise. For every six cells I would hash/check the actual run_config.json, WAV manifest, source/ref files, speed_factor, and instances.log. If those manifests are wrong, stop there; that outranks every modeling hypothesis.

## (b) H1–H3 verdict

H3 — confirmed, but mostly as an instrument bug rather than a clean "v7@1.5 generation bug." The current null statistic is incorrectly normalized for your stated definition, SEGALE grouping is model-dependent, the training slicer contains a genuine missing-span interpolation defect, and there is an unresolved serving/in-process sampler-parity claim. I found no branch in the v7 training pipeline saying "at 1.5× do X," so any true 1.5× failure has to be an interaction rather than a simple speed-conditioned coding error.

H2 — my leading explanation for any residual gap after correcting the metric. The code makes the added corpus extraordinarily boundary-heavy: microturn increments are synthesized as continuous passages, sliced back apart, and every slice receives its own EOS target. Combined with the slicer defect, this is a strong mechanism for changing boundary prosody without changing total audio. The observation "v7 fixes 1× short-turn swallowing but gets worse as evaluation turns become denser" is mechanically consistent with that.

H1 — plausible, but not yet measured. The current checked-in eval is seed-42, not unseeded. If your running experiment simply reruns the command, it may measure essentially no generation-side variance. With truly varied seeds, H1 could still be substantial because five talks and stochastic autoregressive TTS give you a very small system-level sample. As a sanity check, treating 15/291 vs 35/317 as independent Bernoulli blocks gives a naive ≈2.7σ difference—but that calculation is itself invalid because the blocks are model-dependent and clustered, so it neither proves nor rules out H1.

My ordering after the audit is therefore: H3/scoring artifact: confirmed → H2/model-distribution interaction: medium-high → H1/generation variance: unresolved until explicit multi-seed. I would not claim that v7 genuinely doubled fixed-sentence undertranslation at 1.5× yet.

## (c) Measurements ranked by information per hour

| Rank | Measurement | Why it is high-value |
| --- | --- | --- |
| 1 | Recompute fixed-source undertranslation from existing aligned_spacy_hyp.jsonl for all 6 cells. Denominator=input_summary.source_segments; numerator=distinct src_ref_ids in undertranslation blocks; overtranslation separate. Also histogram len(src_ref_ids) for null/non-null. | Minutes, no GPU/ASR. Immediately tells you whether the headline anomaly survives the metric fix. |
| 2 | Verify all six scoring manifests/provenance. Compare run_config.speed_factor, source/ref hashes, WAV manifests/hashes, instances.log, PREFIX, and audio counts. | Minutes. Rules out the hard-coded _speed1/external-scorer hazard before doing science on the output. |
| 3 | Boundary-instrument A/B on the existing WAVs: (a) ASR the whole row as today; (b) cut it at recorded TTS turn durations and ASR each turn independently; (c) alternatively insert 150–300 ms silence at seams and rerun Qwen. | Extremely discriminative. If "missing" text reappears without regenerating any speech, the dominant failure is prosody/ASR/SEGALE coupling, not undertranslation. |
| 4 | Audit v7 training slicing. Count per dataset: turns with zero alignable chars, number of rows with >1 disjoint missing run, 0/1/2-frame targets, boundary repair events, alignment coverage vs turn chars. Patch interpolation to operate on each contiguous missing run and measure how many target cuts move. | Directly quantifies the confirmed v7-amplified code bug. Stratify at ≤5 chars. |
| 5 | Boundary acoustics on current v6/v7 × speeds, especially null-flip sentences: pause/silence around ±300 ms seam, RMS trough, F0 reset, spectral flux, ASR token/timestamp gaps. | Tests H2 directly and uses existing audio. Total duration cannot reveal these effects. |
| 6 | Fix the regen experiment to explicit seeds and run ≥3, preferably 5, seeds/model at 1.5×. Score using corrected fixed-source nulls. Analyze paired by source ID and bootstrap by talk. | This finally measures H1. Do not rely on rerunning seed 42. |
| 7 | Recover the historical v6 serving sampler config. The critical variables are .top_p/.top_k, repetition settings, backend/attention implementation, and seed/RNG behavior. | Cheap if logs survive; instantly resolves the .9/50 parity question. |
| 8 | Matched-engine target A/B: 100–300 representative trajectory groups through old serving and in-process. Compare intended-text ASR recall, forced-align coverage, duration, seam pauses, EOS/code-frame distribution, and downstream sliced frame counts. | Direct answer to question 4; much more informative than arguing that two implementations "should" be equivalent. |
| 9 | Compute actual training weight of trajectory data by supervised audio frames and EOS events, not rows. Include midstart copies; plot turn-char distributions versus eval 1×/1.25×/1.5×. | Tells you how strong the short-turn/EOS prior really is. 12.5k rows may massively understate its loss contribution. |
| 10 | Only after the above: retrain ablations—fixed slicer; trajectory turns coarsened/merged; no midstart; matched-engine targets. | Expensive, but by then each retrain tests a sharply identified mechanism. |

One particularly useful readout for rank 3 is a null-flip table per fixed source sentence: v6 covered/v7 null, v6 null/v7 covered, plus whether per-turn ASR recovers the supposedly missing text. If most of the 1.5× v7-only nulls are audible when ASR is constrained to a turn, you have essentially localized the anomaly to the evaluator. If the text is absent even in per-turn ASR and a second stronger ASR/human spot-check agrees, then H2 becomes a real acoustic/content-generation failure rather than an alignment artifact.

So the next action I would take is not another training run. It is: correct the fixed-unit null metric, verify the speed manifests, and run the no-regeneration per-turn/silence ASR test. Those three should tell you whether there is actually a v7@1.5 undertranslation problem before you spend GPU hours explaining it.
