#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl
from s2s_omni.rasst import audio_duration_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated wav/text outputs against RASST soft-wav dev manifest."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--predictions",
        required=True,
        help="JSONL keyed by id with generated_wav_path and optional prediction/asr_text.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--html", help="Optional audio sample HTML path.")
    parser.add_argument("--max-html", type=int, default=50)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return "".join(str(text or "").split())


def cer(reference: str, candidate: str) -> float | None:
    ref = normalize_text(reference)
    hyp = normalize_text(candidate)
    if not ref:
        return None if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, ref_ch in enumerate(ref, start=1):
        cur = [i]
        for j, hyp_ch in enumerate(hyp, start=1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if ref_ch == hyp_ch else 1),
                )
            )
        prev = cur
    return prev[-1] / len(ref)


def keyed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in rows if row.get("id")}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(rows)}
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if not vals:
            continue
        out[key] = {
            "mean": round(statistics.fmean(vals), 6),
            "p50": percentile(vals, 0.5),
            "p90": percentile(vals, 0.9),
            "min": round(min(vals), 6),
            "max": round(max(vals), 6),
        }
    for key in ["speech_s2s_rtf_violation", "missing_prediction", "early_stop_like"]:
        vals = [row.get(key) for row in rows if key in row]
        if vals:
            out[f"{key}_rate"] = round(sum(1 for value in vals if value) / len(vals), 6)
    return out


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return round(values[0], 6)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return round(values[lo] * (1 - frac) + values[hi] * frac, 6)


def write_html(path: str | Path, rows: list[dict[str, Any]], max_items: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = []
    for row in rows[:max_items]:
        source = row.get("current_source_audio") or ""
        target = row.get("target_wav_path") or ""
        generated = row.get("generated_wav_path") or ""
        items.append(
            f"""
<section>
  <h2>{html.escape(str(row.get('id')))}</h2>
  <p><b>target:</b> {html.escape(str(row.get('target_text') or ''))}</p>
  <p><b>prediction/asr:</b> {html.escape(str(row.get('prediction_text') or row.get('asr_text') or ''))}</p>
  <p>gen {row.get('generated_duration_s')}s / target {row.get('target_duration_s')}s / RTF {row.get('speech_s2s_rtf')}</p>
  <div><span>source</span><br><audio controls src="{html.escape(source)}"></audio></div>
  <div><span>target</span><br><audio controls src="{html.escape(target)}"></audio></div>
  <div><span>generated</span><br><audio controls src="{html.escape(generated)}"></audio></div>
</section>
"""
        )
    path.write_text(
        """
<!doctype html>
<meta charset="utf-8">
<title>RASST Soft-Wav Eval</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;line-height:1.4}
section{border-top:1px solid #ccc;padding:16px 0}
audio{width:520px;max-width:100%}
h2{font-size:16px}
</style>
"""
        + "\n".join(items),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    manifest_rows = read_jsonl(args.manifest)
    pred_by_id = keyed(read_jsonl(args.predictions))
    rows = []
    for gold in manifest_rows:
        row = dict(gold)
        pred = pred_by_id.get(str(gold.get("id")))
        if pred is None:
            row.update({"missing_prediction": True})
            rows.append(row)
            continue
        row.update(pred)
        row["missing_prediction"] = False
        generated_wav = pred.get("generated_wav_path") or pred.get("wav_path")
        if generated_wav:
            row["generated_wav_path"] = generated_wav
            row["generated_duration_s"] = round(audio_duration_s(generated_wav), 6)
            source_duration = float(gold.get("current_source_duration_s") or 0.0)
            if source_duration > 0:
                row["speech_s2s_rtf"] = round(row["generated_duration_s"] / source_duration, 6)
                row["speech_s2s_rtf_violation"] = row["speech_s2s_rtf"] > 1.0
            target_duration = float(gold.get("target_duration_s") or 0.0)
            if target_duration > 0:
                row["generated_target_duration_ratio"] = round(
                    row["generated_duration_s"] / target_duration,
                    6,
                )
                row["early_stop_like"] = row["generated_target_duration_ratio"] < 0.65
        candidate_text = pred.get("asr_text") or pred.get("prediction_text") or pred.get("prediction")
        if candidate_text is not None:
            row["asr_text"] = candidate_text
            value = cer(str(gold.get("target_text") or ""), str(candidate_text))
            if value is not None:
                row["target_asr_cer"] = round(value, 6)
        rows.append(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
    summary = summarize(rows)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.html:
        write_html(args.html, rows, args.max_html)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

