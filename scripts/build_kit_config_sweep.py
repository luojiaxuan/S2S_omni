#!/usr/bin/env python3
"""Compare KIT Lecture Translator configurations on one source clip.

Reads one or more ``run_kit_live_floras.py`` output JSONs (one per KIT config,
e.g. ttsQualityMode=low_latency vs high_quality) and reports, for each:

- stable tts:0 target text and its BLEU(zh)/chrF/CER against a reference,
- tts:0 audio-chunk count and (optionally) resolved target-speech duration,
- TTS timing: first/last tts message arrival relative to stream start, a proxy
  for response latency and end lag.

Outputs a JSON summary and a Markdown table to ``--out-dir``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare KIT configs on one clip.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=RUN_JSON[:TARGET_WAV]",
        help="Config label, run JSON, and optional resolved target wav.",
    )
    parser.add_argument("--reference-file", required=True, help="UTF-8 reference target text.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sacrebleu-path", default="", help="Directory containing sacrebleu.")
    parser.add_argument(
        "--api-key-file",
        default="",
        help="OpenAI API key file. If set, the target wav is transcribed and BLEU "
        "is scored on that ASR transcript (the proper S2S metric) instead of the "
        "KIT tts subtitle text.",
    )
    parser.add_argument("--asr-model", default="gpt-4o-mini-transcribe")
    parser.add_argument("--asr-base-url", default="https://api.openai.com/v1")
    return parser.parse_args()


def asr_transcribe(wav_path: "Path", api_key: str, base_url: str, model: str) -> str:
    import tempfile

    from s2s_omni.openai_asr import transcode_for_upload, transcribe_openai

    with tempfile.TemporaryDirectory() as tmp:
        upload = transcode_for_upload(wav_path, Path(tmp), 25 * 1024 * 1024)
        return transcribe_openai(api_key, base_url, model, upload)


def stable_text(messages: list[dict[str, Any]], key: str = "text") -> str:
    pieces: list[str] = []
    for msg in sorted(messages, key=lambda m: int(m.get("message_id") or 0)):
        if not isinstance(msg, dict) or msg.get("unstable") is True:
            continue
        text = msg.get(key)
        if not isinstance(text, str) or not text.strip() or text.strip() == "TTS-finish":
            continue
        pieces.append(text)
    return "".join(pieces)


def wav_duration_s(path: Path) -> float | None:
    import wave

    if not path.exists():
        return None
    with wave.open(str(path), "rb") as wav:
        return round(wav.getnframes() / wav.getframerate(), 3)


def tts_timing(run: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Latency of tts content relative to the start of the source stream.

    Control messages (session START etc.) are excluded: their
    time_arrive_mediator is the session-creation time, not stream time.
    """
    from datetime import datetime

    started = run.get("startedAt")
    base: float | None = None
    if started:
        base = datetime.fromisoformat(str(started).replace("Z", "+00:00")).timestamp()
    content = [
        m
        for m in messages
        if isinstance(m, dict)
        and not m.get("controll")
        and m.get("time_arrive_mediator")
        and str(m.get("text") or m.get("seq") or "").strip() not in ("", "TTS-finish")
    ]
    arrivals = [float(m["time_arrive_mediator"]) for m in content]
    if not arrivals:
        return {"tts_content_msgs": 0}
    if base is None:
        base = min(arrivals)
    return {
        "tts_first_content_rel_s": round(min(arrivals) - base, 2),
        "tts_last_content_rel_s": round(max(arrivals) - base, 2),
        "tts_content_msgs": len(content),
    }


def compute_metrics(candidate: str, reference: str) -> dict[str, Any]:
    from s2s_omni.floras_live import cer
    from s2s_omni.metrics import optional_sacrebleu, unit_count

    sacre = optional_sacrebleu([candidate], [reference], tokenizer="zh")
    if not sacre.get("available"):
        raise RuntimeError(f"sacreBLEU unavailable: {sacre.get('reason')}")
    return {
        "bleu_zh": round(float(sacre["bleu"]), 2),
        "chrf": round(float(sacre["chrf"]), 2),
        "cer": round(cer(reference, candidate), 4),
        "candidate_units": unit_count(candidate, "zh"),
        "reference_units": unit_count(reference, "zh"),
        "candidate_chars": len(candidate),
    }


def parse_run_spec(spec: str) -> tuple[str, Path, Path | None]:
    label, _, rest = spec.partition("=")
    run_part, sep, wav_part = rest.partition(":")
    # Windows-safe: only treat trailing ``:`` as wav separator when it looks like a path.
    if sep and wav_part:
        return label, Path(run_part).expanduser(), Path(wav_part).expanduser()
    return label, Path(rest).expanduser(), None


def main() -> None:
    args = parse_args()
    if args.sacrebleu_path:
        sys.path.insert(0, str(Path(args.sacrebleu_path).expanduser()))
    reference = Path(args.reference_file).expanduser().read_text(encoding="utf-8").strip()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = ""
    if args.api_key_file:
        api_key = Path(args.api_key_file).expanduser().read_text(encoding="utf-8").strip()

    rows: list[dict[str, Any]] = []
    for spec in args.run:
        label, run_json, target_wav = parse_run_spec(spec)
        run = json.loads(run_json.read_text(encoding="utf-8"))
        components = run.get("collection", {}).get("messagesByComponent", {})
        tts_messages = components.get("tts:0", [])
        tts_text = stable_text(tts_messages, "text")
        row: dict[str, Any] = {
            "label": label,
            "run_json": str(run_json),
            "session_url": run.get("sessionUrl"),
            "source_duration_s": run.get("sourceDurationS"),
            "tts_text": tts_text,
        }
        # KIT subtitle text metrics (a proxy), always computed for transparency.
        tts_metrics = compute_metrics(tts_text, reference)
        row["tts_text_bleu_zh"] = tts_metrics["bleu_zh"]
        row["tts_text_chrf"] = tts_metrics["chrf"]
        row["tts_text_cer"] = tts_metrics["cer"]
        row.update(tts_timing(run, tts_messages))
        if target_wav is not None:
            dur = wav_duration_s(target_wav)
            row["target_wav"] = str(target_wav)
            row["target_duration_s"] = dur
            src = run.get("sourceDurationS")
            if dur is not None and src:
                row["duration_lag_s"] = round(dur - float(src), 2)
        # Primary metric: ASR of the retrieved target audio when a key is given.
        if target_wav is not None and api_key:
            asr_text = asr_transcribe(target_wav, api_key, args.asr_base_url, args.asr_model)
            row["metric_source"] = "asr_of_target_audio"
            row["asr_model"] = args.asr_model
            row["asr_text"] = asr_text
            row["hypothesis_text"] = asr_text
            row.update(compute_metrics(asr_text, reference))
        else:
            row["metric_source"] = "kit_tts_subtitle_text"
            row["hypothesis_text"] = tts_text
            row.update(tts_metrics)
        rows.append(row)

    summary = {
        "reference_file": str(Path(args.reference_file).expanduser()),
        "reference_chars": len(reference),
        "rows": [
            {k: v for k, v in r.items() if k not in ("hypothesis_text", "tts_text", "asr_text")}
            for r in rows
        ],
    }
    (out_dir / "sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    cols = [
        ("label", "config"),
        ("metric_source", "metric src"),
        ("bleu_zh", "BLEU(zh)"),
        ("chrf", "chrF"),
        ("cer", "CER"),
        ("tts_text_bleu_zh", "BLEU tts-text"),
        ("tts_first_content_rel_s", "1st tts s"),
        ("target_duration_s", "target s"),
        ("duration_lag_s", "dur lag s"),
    ]
    lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k, _ in cols) + " |")
    table = "\n".join(lines)
    detail = "\n\n".join(f"### {r['label']}\n\n{r['hypothesis_text']}" for r in rows)
    (out_dir / "sweep_table.md").write_text(
        f"# KIT config sweep (60s FLORAS EN->ZH)\n\n{table}\n\n## Hypotheses\n\n{detail}\n",
        encoding="utf-8",
    )
    print(table)
    print(f"\nwrote {out_dir}/sweep_summary.json and sweep_table.md")


if __name__ == "__main__":
    main()
