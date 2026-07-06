#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.floras_qe import attach_qe_scores, load_qe_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a full FLORAS EN-ZH compare dashboard with KIT rows.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval", action="append", required=True, help="label=eval_dir")
    parser.add_argument("--run-id-prefix", default="en-zh_mono_asr_test__0__speed_")
    parser.add_argument("--qe-scores-jsonl", default="")
    parser.add_argument("--require-qe", action="store_true")
    return parser.parse_args()


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_eval(raw: str) -> tuple[str, Path]:
    label, sep, path = raw.partition("=")
    if not sep or not label.strip():
        raise SystemExit(f"--eval must be label=path, got {raw}")
    return label.strip(), Path(path).expanduser()


def href(path: Path, base: Path) -> str:
    try:
        rel = path.resolve().relative_to(base.resolve())
        return str(rel)
    except ValueError:
        return path.resolve().as_uri()


def link(path: Any, base: Path, label: str) -> str:
    if not path:
        return ""
    target = Path(str(path))
    if not target.exists():
        return ""
    return f'<a href="{esc(href(target, base))}">{esc(label)}</a>'


def audio(path: Any, base: Path) -> str:
    if not path:
        return ""
    target = Path(str(path))
    if not target.exists():
        return ""
    return f'<audio controls preload="metadata" src="{esc(href(target, base))}"></audio>'


def num(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def model_from(label: str, row: dict[str, Any]) -> str:
    text = f"{label} {row.get('compare_backend') or row.get('backend') or ''}".lower()
    if "kit" in text:
        return "kit"
    if "seed" in text:
        return "seed"
    if "gemini" in text:
        return "gemini"
    if "openai" in text or "chatgpt" in text or "gpt" in text:
        return "chatgpt"
    return str(row.get("compare_backend") or row.get("backend") or label)


def chunk_from(label: str, row: dict[str, Any]) -> int | None:
    for key in ("compare_chunk_ms", "chunk_ms"):
        if row.get(key) is not None:
            return int(row[key])
    for marker in ("1920", "960"):
        if marker in label:
            return int(marker)
    return None


def display_chunk(chunk_ms: int | None) -> str:
    return "n/a" if chunk_ms is None else f"{chunk_ms / 1000.0:.2f}s"


def display_variant(label: str, row: dict[str, Any]) -> str:
    label_lower = label.lower()
    if "kit" in label_lower and ("bilang" in label_lower or "bilingual" in label_lower):
        return "mixed-hq-bilingual-no-post-target-asr"
    return str(row.get("display_variant") or "full-target-audio-asr")


def load_metrics(label: str, eval_dir: Path, reference_by_run: dict[str, str]) -> list[dict[str, Any]]:
    paths = [eval_dir / "metrics.jsonl"] if (eval_dir / "metrics.jsonl").exists() else sorted(eval_dir.glob("*/metrics.json"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        raw_rows = read_jsonl(path) if path.name.endswith(".jsonl") else [read_json(path)]
        for row in raw_rows:
            run_id = str(row.get("run_id") or "")
            if not run_id:
                continue
            model = model_from(label, row)
            chunk_ms = chunk_from(label, row)
            run_dir = Path(str(row.get("run_dir") or path.parent / run_id))
            generated = row.get("generated_audio_path") or row.get("generated_wav_path") or run_dir / "generated_target.wav"
            source_stream = (
                row.get("source_stream_audio_path")
                or row.get("source_stream_wav_path")
                or run_dir / "source_stream_24k.wav"
            )
            if not Path(str(source_stream)).exists():
                source_stream = run_dir / "source_stream_16k.wav"
            row = dict(row)
            row.update(
                {
                    "eval_label": label,
                    "display_model": model,
                    "display_chunk": display_chunk(chunk_ms),
                    "display_variant": display_variant(label, row),
                    "compare_chunk_ms": chunk_ms,
                    "run_dir": str(run_dir),
                    "run_page_path": str(row.get("run_page") or row.get("run_page_path") or run_dir / "index.html"),
                    "generated_audio_path": str(generated),
                    "source_stream_audio_path": str(source_stream),
                    "reference_text": reference_by_run.get(run_id, ""),
                }
            )
            rows.append(row)
    return rows


def row_speed(row: dict[str, Any]) -> float:
    return float(row.get("speed_factor") or 0.0)


def sort_key(row: dict[str, Any]) -> tuple[float, int, int, str]:
    order = {"chatgpt": 0, "gemini": 1, "seed": 2, "kit": 3}
    return (
        row_speed(row),
        order.get(str(row.get("display_model")), 99),
        int(row.get("compare_chunk_ms") or 999999),
        str(row.get("eval_label")),
    )


def row_dict(row: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "run_id",
        "display_model",
        "display_chunk",
        "display_variant",
        "eval_label",
        "speed_factor",
        "source_eval_duration_s",
        "source_stream_duration_s",
        "generated_duration_s",
        "full_s2s_rtf",
        "end_lag_s",
        "wall_clock_end_delay_s",
        "max_backlog_s",
        "max_playback_queue_s",
        "bleu",
        "chrf",
        "cer",
        "xcomet_qe_score",
        "metricx_qe_score",
        "metricx_qe_error",
        "xcomet_qe_model",
        "metricx_qe_model",
        "qe_segment_count",
        "qe_reference_free",
        "qe_segmentation",
        "candidate_text_source",
        "generated_audio_path",
        "source_stream_audio_path",
        "run_page_path",
    ]
    return {key: row.get(key) for key in keep}


def metric_rows_text_sources(rows: list[dict[str, Any]]) -> str:
    sources = sorted({str(row.get("candidate_text_source") or "") for row in rows if row.get("candidate_text_source")})
    return ", ".join(sources)


def render_table(rows: list[dict[str, Any]], out_dir: Path) -> str:
    pieces: list[str] = []
    last_speed = ""
    for row in sorted(rows, key=sort_key):
        speed = f"{row_speed(row):g}"
        if speed != last_speed:
            pieces.append(f'<tr class="speed-group"><td colspan="21">speed={esc(speed)}</td></tr>')
            last_speed = speed
        pieces.append(
            "<tr>"
            f"<td>{esc(speed)}</td>"
            f"<td>{esc(row.get('display_model'))}</td>"
            f"<td>{esc(row.get('display_chunk'))}</td>"
            f"<td>{esc(row.get('display_variant'))}</td>"
            f"<td>{num(row.get('xcomet_qe_score'), 4)}</td>"
            f"<td>{num(row.get('metricx_qe_score'), 3)}</td>"
            f"<td>{num(row.get('metricx_qe_error'), 3)}</td>"
            f"<td>{num(row.get('bleu'))}</td>"
            f"<td>{num(row.get('chrf'))}</td>"
            f"<td>{num(row.get('cer'), 3)}</td>"
            f"<td>{num(row.get('source_stream_duration_s'))}</td>"
            f"<td>{num(row.get('generated_duration_s'))}</td>"
            f"<td>{num(row.get('full_s2s_rtf'), 3)}</td>"
            f"<td>{num(row.get('end_lag_s'))}</td>"
            f"<td>{num(row.get('wall_clock_end_delay_s'))}</td>"
            f"<td>{num(row.get('max_backlog_s'))}</td>"
            f"<td>{num(row.get('max_playback_queue_s'))}</td>"
            f"<td>{esc(row.get('candidate_text_source'))}</td>"
            f"<td>{link(row.get('run_page_path'), out_dir, 'open')}</td>"
            f"<td>{audio(row.get('source_stream_audio_path'), out_dir)}</td>"
            f"<td>{audio(row.get('generated_audio_path'), out_dir)}</td>"
            "</tr>"
        )
    return "\n".join(pieces)


def render_details(rows: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    for row in sorted(rows, key=sort_key):
        title = (
            f"speed={row_speed(row):g} / {row.get('display_model')} / "
            f"{row.get('display_chunk')} / BLEU {num(row.get('bleu'))}"
        )
        pieces.append(
            f"<details><summary>{esc(title)}</summary>"
            f"<div class=\"textgrid\"><div><h3>Hypothesis</h3><pre>{esc(row.get('candidate_text'))}</pre></div>"
            f"<div><h3>Reference</h3><pre>{esc(row.get('reference_text'))}</pre></div></div>"
            "</details>"
        )
    return "\n".join(pieces)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_by_run = {
        str(row.get("run_id")): str(row.get("target_reference_text") or "")
        for row in read_jsonl(Path(args.manifest).expanduser())
    }
    rows: list[dict[str, Any]] = []
    for raw in args.eval:
        label, eval_dir = parse_eval(raw)
        rows.extend(row for row in load_metrics(label, eval_dir, reference_by_run) if str(row.get("run_id", "")).startswith(args.run_id_prefix))
    qe_scores = load_qe_scores(args.qe_scores_jsonl)
    rows = [attach_qe_scores(row, qe_scores) for row in rows]
    if args.require_qe:
        missing_qe = [
            row
            for row in rows
            if row.get("xcomet_qe_score") is None
            or row.get("metricx_qe_score") is None
            or row.get("xcomet_qe_segments") != row.get("qe_segment_count")
            or row.get("metricx_qe_segments") != row.get("qe_segment_count")
            or row.get("qe_hypothesis_chars_mismatch") is not None
        ]
        if missing_qe:
            labels = ", ".join(
                f"{row.get('run_id')}:{row.get('eval_label')}" for row in missing_qe[:8]
            )
            raise SystemExit(f"--require-qe found {len(missing_qe)} rows without QE scores: {labels}")
    rows = sorted(rows, key=sort_key)
    summary = {
        "rows": [row_dict(row) for row in rows],
        "row_count": len(rows),
        "reference_sha256_by_run": {
            run_id: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for run_id, text in reference_by_run.items()
            if run_id.startswith(args.run_id_prefix)
        },
    }
    (out_dir / "compare_metrics.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_doc = f"""<!doctype html>
<meta charset="utf-8">
<title>FLORAS EN-ZH Full Live S2S Compare</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;color:#202428;background:#fbfbfc}}
h1{{font-size:22px;margin:0 0 8px}}.meta{{color:#667085;margin:0 0 18px;line-height:1.45}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{border-top:1px solid #d6d9de;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{background:#f2f4f7;font-weight:600}}audio{{width:210px}}details{{margin:14px 0;padding:12px;background:white;border:1px solid #d6d9de;border-radius:6px}}
summary{{cursor:pointer;font-weight:600}}.speed-group td{{background:#e8eef7;font-weight:700;color:#344054;border-top:2px solid #aeb8c7}}
.textgrid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}}pre{{white-space:pre-wrap;word-break:break-word;font-size:13px;line-height:1.45;background:#f8fafc;padding:10px;border-radius:6px}}
h3{{font-size:13px;margin:0 0 6px}}@media(max-width:900px){{.textgrid{{grid-template-columns:1fr}}table{{display:block;overflow-x:auto}}}}
</style>
<h1>FLORAS EN-ZH Full Live S2S Compare</h1>
<p class="meta">
{len(rows)} full-source rows over the same 1072.63s FLORAS EN-&gt;ZH source content. speed=1.5 streams the same source content in about 715s.
The text source column records how each hypothesis was produced; KIT rows use target-speech ASR and not backend subtitle text.
KIT currently has only the 1.92s chunk row in this full dashboard. BLEU uses the metrics already stored in each eval run, and hypothesis/reference text is shown with punctuation preserved.
Observed text sources: {esc(metric_rows_text_sources(rows))}.
</p>
<p class="meta">
Timing metrics are read from each eval run's audio chunk timeline. KIT timing uses retrieved <code>tts:0</code> target-audio chunk arrival times, then the same FLORAS evaluator computes duration lag, wall delay, backlog, and playback queue.
QE columns are reference-free source+hypothesis scores over proportional text chunks; xCOMET-QE and MetricX-QE are higher-is-better, while MetricX err is lower-is-better.
The current QE score file covers all rows in this dashboard; rebuild with <code>--require-qe</code> to catch missing or stale QE rows.
</p>
<table>
<thead><tr>
<th>speed</th><th>model</th><th>chunk</th><th>variant</th>
<th>xCOMET-QE ↑</th><th>MetricX-QE ↑</th><th>MetricX err ↓</th><th>BLEU zh</th><th>chrF</th><th>CER</th>
<th>stream s</th><th>target s</th><th>RTF</th><th>duration lag</th><th>wall delay</th><th>max backlog</th><th>max queue</th>
<th>text source</th><th>detail</th><th>source audio</th><th>target audio</th>
</tr></thead>
<tbody>
{render_table(rows, out_dir)}
</tbody>
</table>
<h2>Raw Hypothesis / Reference Text</h2>
{render_details(rows)}
"""
    (out_dir / "index.html").write_text(html_doc, encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output_dir": str(out_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
