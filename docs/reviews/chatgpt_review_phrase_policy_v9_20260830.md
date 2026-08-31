# ChatGPT 外审全文：v9 write policy 设计（2026-08-30）

- 通道：网页 ChatGPT 临时聊天，推理档「极高」，思考 1m 33s，回复中做了网页检索。
- 发审前的自我判断已冻结于 [`prior_judgment_phrase_policy_v9_20260830.md`](./prior_judgment_phrase_policy_v9_20260830.md)。

## 提交的请求（原文）

SETUP: streaming speech-to-speech cascade. A simultaneous speech translation model
emits target text incrementally on a fixed audio chunk grid (chunk = multiplier x
0.96s); its text feeds a streaming TTS. (1) CURRENT WRITE POLICY: accumulate across
chunks; write only when the buffer ends with intra-sentence punctuation AND has >=6
content chars; force-write after 7.68s. (2) PROPOSED: drop the punctuation condition
and the force-write cap; control latency only by chunk size (multiplier 1..4); keep
one sweepable floor min_words, default 2. Rationale: fewer heuristics, one physically
meaningful knob. (3) MEASURED on 6385 real trajectories plus a 5-talk cascade eval:
13.4% (mult=1) and 25.8% (mult=2) of current writes are force-cut and end mid-phrase,
so the phrase-boundary premise fails on a quarter of the supervision; downstream TTS
character error rate falls monotonically with write length, 37.4% at 1-5 chars down to
7.7% at 31+ chars; but TTS over-generation rate peaks at 16-30 chars (8.5%) and is NOT
monotonic; the 4.5 BLEU drop at 1.5x source speed is caused by over-generation doubling
(6.0% to 11.4% of turns), not by worse text. (4) MY VIEW: adopt it, but sweep multiplier
x min_words jointly since length control really lives in multiplier, and do not use BLEU
alone as the sweep criterion because over-generation is non-monotonic in length.
QUESTIONS: (a) three weakest points a reviewer would attack; (b) existing benchmarks or
published write policies I should compare against; (c) if you disagree, say why and what
you would do instead. Be blunt.

## 回复全文

I mostly agree with adopting it, but I would not ship the proposal exactly as stated.
The right simplification is “remove punctuation as a gating heuristic”; the risky part
is “remove the latency cap entirely.”

### (a) The three weakest points a reviewer will attack

**Your force-cut statistic does not actually falsify the punctuation premise.** The
current rule explicitly says “prefer punctuation, but violate that preference after
7.68 s.” So observing that force-writes end mid-phrase is almost tautological. The 25.8%
number at multiplier 2 is strong evidence that the 7.68 s backstop is operating as a
normal path rather than a backstop, but it does not establish that punctuation-conditioned
writes are bad when punctuation is available.

The missing experiment is a matched policy ablation: for the same incremental translation
trajectories, replay TTS under punctuation-gated vs punctuation-free segmentation while
controlling for write length. Your 6,385 stored trajectories are ideal for this. You can
isolate the batching policy without changing translation output. If non-punctuation writes
at matched length are no worse—or better—on TTS fidelity/fluency, then you have killed the
phrase-boundary hypothesis.

**“Chunk size is the one physical latency knob” is too strong and probably false for S2S.**
Chunk size controls when you get an opportunity to write; it does not uniquely determine
audible latency. min_words is already another control variable, and translation emission
rate, target-language expansion, TTS compute, synthesized speech duration, and playback
backlog all matter. This is especially relevant given your own over-generation result:
extra generated audio directly creates “audio debt,” which can accumulate downstream even
if text arrives promptly. Earlier S2S work explicitly found that source/target speech-rate
mismatch can accumulate latency over successive sentences, motivating adaptive control
rather than relying only on translation chunk timing.

So I would change the claim to: “chunk multiplier is the primary scheduling knob;
min_words is a small-fragment guard.” That is defensible. “Only latency knob” invites an
easy counterexample.

**Your length evidence is observational, not causal—and the non-monotonic over-generation
result makes this more important.** CER falling from 37.4% to 7.7% with length strongly
motivates avoiding tiny writes. But the 16–30-character over-generation peak means “longer
is better” is already false for one failure mode. Length bins also co-vary with punctuation,
lexical content, prosody, sentence position, model confidence, and speech rate. The
five-talk end-to-end result is particularly vulnerable to a “small eval / particular talks”
criticism.

I would explicitly avoid claiming that the observed curves identify an optimum. Say they
motivate a policy-level factorial sweep. And report insertion/over-generation separately
from aggregate CER/WER; otherwise an apparently better character error rate can conceal
exactly the hallucination mode hurting BLEU.

### (b) What I would compare against

The closest published comparator is not just wait-k. There is a well-established family of
stable-hypothesis/commit policies: **Hold-n, Local Agreement (LA-n), and Shared Prefix
(SP-n)**. CUNI-KIT evaluated these while separately varying speech chunk size, and
subsequent work showed Local Agreement/Hold-n can be used specifically to control the
quality–latency tradeoff of incremental speech translation. CMU’s IWSLT 2024 system used a
fixed Hold-7 policy with chunked speech; NAIST’s 2024 system used Local Agreement for its
submitted SimulST system.

If you have access to revisable translation hypotheses before commitment, **LA-2 is the
baseline I would most want in the paper.** It tests a much more principled version of the
thing your punctuation rule was trying to approximate: “is this prefix stable enough to
hand downstream?” If your translator only exposes irrevocably emitted text, say so
explicitly; then LA/SP are not implementable at the TTS boundary, and that strengthens the
case for your simpler batching rule.

For broader positioning, include wait-k as the canonical fixed policy and **AlignAtt/EDAtt**
as adaptive alternatives. Wait-k established the standard controllable fixed-lag
formulation, while AlignAtt/EDAtt make emission decisions using audio–translation attention
rather than punctuation or raw elapsed time.

There is also a very direct TTS precedent: **incremental prefix-to-prefix TTS has
demonstrated synthesis with roughly 1–2 words of textual lookahead**, which makes your
min_words=2 choice much easier to motivate as a starting value rather than an arbitrary
magic number.

For benchmarks, I would use IWSLT simultaneous S2S 2023/2024 conventions wherever possible.
IWSLT 2024 used ASR-BLEU for simultaneous S2S, reported AL/LAAL/ATD based on aligned
transcripts, and used a 2.5 s starting-offset constraint for S2S. The 2023 campaign
additionally reported BLASER, start offset, end offset, and ATD for S2S systems.

For the long-form/stall story, borrow from the newer evaluation direction too. IWSLT 2026
moved simultaneous ST to unsegmented long-form audio, uses **LongYAAL** as its primary
latency measure and **StreamLAAL** for continuity with prior work, and evaluates quality
primarily with COMET-XL. Recent latency work has also argued that averages hide large
tail-latency excursions. That is directly relevant if you eliminate the force cap.

### (c) What I would actually implement

I would remove punctuation from the normal write decision and do the joint multiplier ×
min_words sweep you propose. But I would **retain a very loose starvation watchdog**, not as
a tuned latency mechanism but as a reliability invariant. The distinction matters.

Your present 7.68 s cap is plainly too active—13.4–25.8% activation means it is part of the
operating policy. Instead, derive a watchdog from the no-write-gap distribution of the new
policy, e.g. somewhere beyond the empirical p99/p99.9 or from an explicit product SLA. In a
healthy configuration it should almost never fire. Then you can truthfully say that ordinary
latency is governed by the chunk schedule while still preventing a pathological
one-word/zero-word emission sequence from producing arbitrarily long silence.

I would also **not choose the sweep winner by a single scalar metric**. Use a
Pareto/constraint formulation. First reject configurations with unacceptable
over-generation, TTS insertion rate, or tail latency. Among the feasible ones, choose the
best translation/speech quality at the desired latency. In particular, report
intended-TTS-text → synthesized-ASR substitution/deletion/insertion separately, because
your result suggests insertion/over-generation is the failure that actually matters at high
source rate.

The latency panel I would want is: start offset, end offset or ATD/LongYAAL where
applicable, p50/p95/p99 time between audible writes, and **maximum/playback backlog
(“seconds of synthesized audio queued”)**. That last quantity may explain the 1.5×
degradation more directly than text latency does. For quality, keep upstream text
BLEU/COMET to demonstrate that translation itself did not degrade, then ASR-BLEU or
ASR-COMET/BLASER on speech, TTS CER/WER with error decomposition, over-generation rate, and
a small MOS/fluency evaluation if this is headed for publication.

So the thesis I would defend is slightly narrower than yours:

> Punctuation is a poor commit gate for this cascade, and the existing timeout fires too
> often to function as a mere safeguard. A fixed audio-chunk schedule plus a minimal
> output-size floor gives a simpler and more controllable operating policy; a
> non-operational starvation watchdog remains only for pathological tails. The operating
> point must be selected jointly for text-batching quality, TTS over-generation, and
> end-to-end latency rather than BLEU alone.

That version is stronger because your measurements actually support every part of it.
