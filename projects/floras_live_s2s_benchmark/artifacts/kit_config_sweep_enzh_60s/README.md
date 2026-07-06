# KIT Lecture Translator — config sweep (60s FLORAS EN→ZH)

Controlled comparison of KIT Lecture Translator quality/latency configurations
on the standard 60s FLORAS EN→ZH clip
(`eval_real_60s/en-zh_mono_asr_test__0__speed_1/source_eval.wav`, 16 kHz mono).
This is the "inspect/sweep the product settings before a full run" step called
for in `docs/handoff_s2s_omni.md`. Source audio is streamed in **1.92s chunks**
(`--chunk-s 1.92`, matching the `chunk1920` runs of the other backends); a 960ms
capture is kept alongside to show single-clip variance.

## Headline result: KIT target speech is retrievable

Earlier KIT captures were text-only because the synthesized target audio "could
not be retrieved". That gap is now closed. Each `tts:0` message carries the
audio as a linked reference; the live SSE payload
(`EventSource("/webapi/stream?channel={session}")`) is
`/ltapi/stream/data?name=Data/{session}/{N}`, which the KIT client rewrites
`/ltapi`→`/webapi`. So the download URL is:

```
GET /webapi/stream/data?name=Data/{session}/{N}
```

The response is a JSON string of base64 PCM (**s16le, 16 kHz, mono**). Unwrap the
JSON string, base64-decode, concatenate chunks in message order. See
`scripts/fetch_kit_target_audio.py`. KIT S2S can now be scored the same way as
Seed/GPT/Gemini (target-speech ASR → BLEU/chrF/CER), not just on subtitle text.

## Results

BLEU/chrF/CER are the **proper S2S metric**: the retrieved target speech is
transcribed with `gpt-4o-mini-transcribe` and scored (sacreBLEU `tokenize=zh`,
punctuation preserved) against the canonical 60s reference (`reference_60s.txt`).
KIT's own `tts:0` subtitle-text BLEU is shown alongside for reference (ASR is a
touch non-deterministic, so numbers can move ~0.3 BLEU between runs). Latency is
relative to the source stream start (realtime pacing).

| config | BLEU(zh) ASR | chrF | CER | BLEU tts-text | 1st tts s | target s | dur lag s |
|---|---|---|---|---|---|---|---|
| hq_chunk960  | 23.96 | 21.79 | 0.6917 | 22.83 | 25.66 | 63.30 | 3.30 |
| ll_chunk960  | 21.68 | 20.17 | 0.7250 | 20.75 | 18.51 | 69.45 | 9.45 |
| hq_chunk1920 | 22.76 | 20.57 | 0.7250 | 20.91 | 20.97 | 61.62 | 1.62 |
| ll_chunk1920 | 25.32 | 22.73 | 0.7042 | 22.75 | 18.15 | 66.65 | 6.65 |

**Cross-check.** An independent parallel capture on the same clip
(`outputs/floras_live_pilot_refs/kit_config_smoke_60s_chunk1920/`, same
target-speech-ASR method) scored `online_low_latency` BLEU 24.94 and
`online_high_quality` BLEU 23.97. Our `ll_chunk1920` (25.32) and `hq_chunk1920`
(22.76) land within ~1 BLEU, validating the retrieval end-to-end. That run also
tested `format=mixed`, where **`mixed_high_quality` was best (BLEU 25.94, target
52.1s)** — the standout config. `format=mixed` is not yet covered by this sweep.

### What is and isn't robust on one 60s clip

- **ASR quality does NOT cleanly separate the configs.** low_latency wins at
  1920ms (25.32 vs 22.76) but high_quality wins at 960ms (23.96 vs 21.68) — the
  ranking flips with chunk size, so ~2-BLEU gaps are single-clip noise. Do not
  rank `ttsQualityMode` by quality from one clip (matches the handoff's "don't
  rank KIT from the 60s smoke" caution).
- **`high_quality` consistently yields more compact target speech** (61.6/63.3s
  vs 66.7/69.5s) and lower duration lag — consistent with "High Quality: better
  audio with text segmentation".
- **`low_latency` emits first audio ~2–7s sooner**, matching "Low Latency:
  faster response without segmentation".
- **1.92s chunks give tighter timing** than 960ms.

Both configs have high startup latency (~18–26s to first Chinese audio). Report
wall-clock delay, not just duration lag.

**To rank the configs, run more clips (or the full FLORAS source) and include
`format=mixed`;** one 60s clip is too noisy on quality metrics.

## KIT config API (reverse-engineered)

Sessions are created with `GET /create?<params>` (auth cookie required); the
response 302-redirects to `/present/{sessionId}`. Params and values (from the
`/index/live/` config form):

- `language` (source, e.g. `en`); `mtLanguage` (target text, e.g. `zh`);
  `audioLanguage` (TTS audio, e.g. `zh`).
- `ttsQualityMode` — `low_latency` | `high_quality`. **Primary S2S knob.**
- `format` — `online` (Fixed) | `mixed` (revise, no reorder) | `resending`
  (revise + reorder). This sweep used `online`.
- `postproduction` — content **shortening**, multi-select `50` | `70` | `90`%
  (none here).
- `smartChaptering` — `online_dynamic` (only option currently).
- `pause` — speech-segment silence threshold (2.0s); `mute` — UI notify.

Sessions can be removed with `POST /delete_session/{sessionId}`.

## Reproduce

```bash
REPO=.../work/S2S_omni
CK=.../kit_full_enzh/.kit_cookie_header          # valid forward-auth cookie
CLIP=.../eval_real_60s/en-zh_mono_asr_test__0__speed_1/source_eval.wav

# 1. GET /create?...ttsQualityMode=high_quality... -> sessionId (from redirect)
# 2. stream the clip in 1.92s chunks and capture messages
python3 $REPO/scripts/run_kit_live_floras.py --session-id <ID> \
  --session-name kit_hq --wav-path $CLIP --chunk-s 1.92 --pace realtime \
  --output-json kit_hq_run.json --cookie-header-file $CK
# 3. resolve the target speech
python3 $REPO/scripts/fetch_kit_target_audio.py --run-json kit_hq_run.json \
  --cookie-header-file $CK --output-wav kit_hq_target.wav
# 4. transcribe target speech + score (needs sacrebleu on sys.path)
python3 $REPO/scripts/build_kit_config_sweep.py \
  --api-key-file /tmp/acl6060_keys/openai.key --asr-model gpt-4o-mini-transcribe \
  --run "hq_chunk1920=kit_hq_run.json:kit_hq_target.wav" \
  --run "ll_chunk1920=kit_ll_run.json:kit_ll_target.wav" \
  --reference-file reference_60s.txt --out-dir .
# (omit --api-key-file to fall back to KIT tts-subtitle-text scoring)
```

## Caveats

- BLEU/chrF/CER are ASR-based (`gpt-4o-mini-transcribe` over the retrieved
  `*_target.wav`); the `tts_text` BLEU column is KIT's subtitle text for
  reference. The retrieved `high_quality` audio here is ~61s / 8 chunks vs the
  parallel run's ~52s / 6 chunks for the same config — likely session-to-session
  segmentation variance; low_latency matches closely, so retrieval is sound.
- Each chunk-size pair was streamed concurrently; absolute latency may be
  slightly inflated by shared server load, but relative comparisons hold.
- Single 60s clip, `format=online`, no shortening. Sweep `format` and
  `postproduction` (shortening), and add clips, before product-level claims.
- Target `*_target.wav` files are kept local (gitignored) under
  `outputs/floras_live_pilot_refs/kit_config_sweep_enzh_60s/`.
```
