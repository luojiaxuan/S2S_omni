# KIT Lecture Translator — config sweep (60s FLORAS EN→ZH)

Controlled comparison of KIT Lecture Translator quality/latency configurations
on the standard 60s FLORAS EN→ZH clip
(`eval_real_60s/en-zh_mono_asr_test__0__speed_1/source_eval.wav`, 16 kHz mono).
This is the "inspect/sweep the product settings before a full run" step called
for in `docs/handoff_s2s_omni.md`.

## Headline result: KIT target speech is retrievable

Earlier KIT captures were text-only because the synthesized target audio "could
not be retrieved". That gap is now closed. Each `tts:0` message carries the
audio as a linked reference; the live SSE payload is
`/ltapi/stream/data?name=Data/{session}/{N}` and the download URL is:

```
GET /webapi/stream/data?name=Data/{session}/{N}
```

The response is a JSON string of base64 PCM (**s16le, 16 kHz, mono**). See
`scripts/fetch_kit_target_audio.py`. KIT S2S can now be scored the same way as
Seed/GPT/Gemini (target-speech ASR → BLEU/chrF/CER), not just on subtitle text.

## Results

Text metrics use sacreBLEU `tokenize=zh` with punctuation preserved, scored on
KIT's own `tts:0` text against the canonical 60s reference
(`reference_60s.txt`). Latency columns are relative to the source stream start
(realtime pacing).

| config | BLEU(zh) | chrF | CER | hyp chars | 1st tts s | last tts s | target s | dur lag s |
|---|---|---|---|---|---|---|---|---|
| high_quality | 22.83 | 21.28 | 0.7042 | 295 | 25.66 | 86.23 | 63.30 | 3.30 |
| low_latency  | 20.75 | 19.65 | 0.7333 | 283 | 18.51 | 86.92 | 69.45 | 9.45 |

`ttsQualityMode=high_quality` ("better audio quality with text segmentation")
produced better translation quality **and** more compact, less-laggy target
speech (63.3s vs 69.45s for a 60s source), at ~7s higher first-token latency.
`low_latency` ("faster response without segmentation") emitted first audio
sooner but was choppier and longer. **Recommendation: use `high_quality` for a
quality-focused full FLORAS run**; pick `low_latency` only if minimizing
first-token latency matters more than quality.

Both configs have a high initial latency (~18–26s to first Chinese audio) and
end ~26s after the source stream — the cascade (ASR→MT→segment→TTS) buffers
substantially. Report wall-clock delay, not just duration lag, in any full run.

## KIT config API (reverse-engineered)

Sessions are created with `GET /create?<params>` (auth cookie required); the
response 302-redirects to `/present/{sessionId}`. Relevant params and values
(from the `/index/live/` config form):

- `language` — source (e.g. `en`); `mtLanguage` — target text (e.g. `zh`);
  `audioLanguage` — TTS audio language (e.g. `zh`).
- `ttsQualityMode` — `low_latency` | `high_quality`. **Primary S2S knob.**
- `format` — `online` (Fixed) | `mixed` (revise, no reorder) | `resending`
  (revise + reorder). This sweep used `online`.
- `postproduction` — content **shortening**, multi-select `50` | `70` | `90`%
  (none here).
- `smartChaptering` — `online_dynamic` (only option currently).
- `pause` — speech-segment silence threshold (2.0s here); `mute` — UI notify.

## Reproduce

```bash
REPO=.../work/S2S_omni
CK=.../kit_full_enzh/.kit_cookie_header          # valid forward-auth cookie
CLIP=.../eval_real_60s/en-zh_mono_asr_test__0__speed_1/source_eval.wav

# 1. create a session (GET /create ... ttsQualityMode=high_quality) -> sessionId
# 2. stream the clip and capture messages
python3 $REPO/scripts/run_kit_live_floras.py --session-id <ID> \
  --session-name kit_hq --wav-path $CLIP \
  --output-json kit_hq_run.json --cookie-header-file $CK --pace realtime
# 3. resolve the target speech
python3 $REPO/scripts/fetch_kit_target_audio.py --run-json kit_hq_run.json \
  --cookie-header-file $CK --output-wav kit_hq_target.wav
# 4. score / compare (needs sacrebleu on sys.path)
python3 $REPO/scripts/build_kit_config_sweep.py \
  --run "high_quality=kit_hq_run.json:kit_hq_target.wav" \
  --run "low_latency=kit_ll_run.json:kit_ll_target.wav" \
  --reference-file reference_60s.txt --out-dir .
```

## Caveats

- Text metrics are on KIT's `tts:0` subtitle text, a proxy for the spoken
  target. For a rigorous S2S score, ASR the retrieved `*_target.wav`
  (gpt-4o-mini-transcribe) and recompute BLEU/chrF/CER — the audio is now
  available for this.
- The two configs were streamed concurrently; absolute latency may be slightly
  inflated by shared server load, but the relative comparison holds.
- Single 60s clip, `format=online`, no shortening. Sweep `format` and
  `postproduction` (shortening) before drawing product-level conclusions.
- Target `*_target.wav` files are kept local (gitignored) under
  `outputs/floras_live_pilot_refs/kit_config_sweep_enzh_60s/`.
```
