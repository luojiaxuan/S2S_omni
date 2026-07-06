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

Text metrics use sacreBLEU `tokenize=zh` (punctuation preserved) on KIT's own
`tts:0` text vs the canonical 60s reference (`reference_60s.txt`). Latency is
relative to the source stream start (realtime pacing).

| config | BLEU(zh) | chrF | CER | hyp chars | 1st tts s | last tts s | target s | dur lag s |
|---|---|---|---|---|---|---|---|---|
| hq_chunk960  | 22.83 | 21.28 | 0.7042 | 295 | 25.66 | 86.23 | 63.30 | 3.30 |
| ll_chunk960  | 20.75 | 19.65 | 0.7333 | 283 | 18.51 | 86.92 | 69.45 | 9.45 |
| **hq_chunk1920** | 20.91 | 19.39 | 0.7417 | 284 | 20.97 | 79.80 | 61.62 | 1.62 |
| **ll_chunk1920** | 22.75 | 20.73 | 0.7208 | 276 | 18.15 | 79.06 | 66.65 | 6.65 |

### What is and isn't robust on one 60s clip

- **Text quality does NOT separate the two configs.** `high_quality` wins at
  960ms chunks but `low_latency` wins at 1920ms chunks; the ~2-BLEU gaps are
  within single-clip noise. Do not rank `ttsQualityMode` by quality from this
  clip — that matches the handoff's "don't rank KIT from the 60s smoke" caution.
- **`high_quality` consistently yields more compact target speech** (61.6/63.3s
  vs 66.7/69.5s) and lower duration lag (1.6/3.3s vs 6.7/9.5s). This is the one
  stable difference — consistent with "High Quality: better audio with text
  segmentation".
- **`low_latency` emits first audio ~2–7s sooner** (18.1–18.5s vs 21.0–25.7s),
  matching "Low Latency: faster response without segmentation".
- **1.92s chunks give tighter timing** than 960ms (lower duration lag and
  first-token latency for both configs).

Both configs have high startup latency (~18–26s to first Chinese audio) and end
~19–26s after the source stream. Report wall-clock delay, not just duration lag.

**To actually rank the configs, run more clips (or the full FLORAS source);** a
single 60s clip is too noisy on text metrics.

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
# 4. score / compare (needs sacrebleu on sys.path)
python3 $REPO/scripts/build_kit_config_sweep.py \
  --run "hq_chunk1920=kit_hq_run.json:kit_hq_target.wav" \
  --run "ll_chunk1920=kit_ll_run.json:kit_ll_target.wav" \
  --reference-file reference_60s.txt --out-dir .
```

## Caveats

- Text metrics are on KIT's `tts:0` subtitle text, a proxy for the spoken
  target. For a rigorous S2S score, ASR the retrieved `*_target.wav`
  (gpt-4o-mini-transcribe) and recompute BLEU/chrF/CER — the audio is now
  available for this.
- Each chunk-size pair was streamed concurrently; absolute latency may be
  slightly inflated by shared server load, but relative comparisons hold.
- Single 60s clip, `format=online`, no shortening. Sweep `format` and
  `postproduction` (shortening), and add clips, before product-level claims.
- Target `*_target.wav` files are kept local (gitignored) under
  `outputs/floras_live_pilot_refs/kit_config_sweep_enzh_60s/`.
```
