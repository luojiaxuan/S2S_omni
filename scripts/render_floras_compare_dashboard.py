#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.floras_live import esc, rel
from s2s_omni.floras_qe import attach_qe_scores, load_qe_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a cross-backend FLORAS live dashboard.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="FLORAS Live S2S Compare")
    parser.add_argument(
        "--eval",
        action="append",
        required=True,
        help="Repeated label=eval_dir entries, e.g. openai_960=/path/to/eval.",
    )
    parser.add_argument("--run-id-prefix", default="")
    parser.add_argument("--run-id-regex", default="")
    parser.add_argument("--qe-scores-jsonl", default="")
    parser.add_argument("--require-qe", action="store_true")
    return parser.parse_args()


def parse_eval_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise SystemExit(f"--eval must be label=path, got: {raw}")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"--eval label cannot be empty: {raw}")
    return label, Path(path).expanduser()


def infer_backend(label: str, row: dict[str, Any]) -> str:
    text = label.lower()
    if "seed" in text:
        return "seed"
    if "gemini" in text:
        return "gemini"
    if "openai" in text or "chatgpt" in text or "gpt" in text:
        return "chatgpt"
    backend = str(row.get("backend") or "").lower()
    if "gemini" in backend:
        return "gemini"
    if "openai" in backend:
        return "chatgpt"
    return label


def infer_chunk_ms(label: str, row: dict[str, Any]) -> int | None:
    if row.get("chunk_ms") is not None:
        try:
            return int(row["chunk_ms"])
        except (TypeError, ValueError):
            pass
    match = re.search(r"(?:chunk)?(\d{3,5})", label)
    if match:
        return int(match.group(1))
    return None


def load_metrics(eval_label: str, eval_dir: Path) -> list[dict[str, Any]]:
    metrics_path = eval_dir / "metrics.jsonl"
    rows: list[dict[str, Any]] = []
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    else:
        for path in sorted(eval_dir.glob("*/metrics.json")):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    for row in rows:
        run_id = str(row.get("run_id") or "")
        generated_path = existing_path(row.get("generated_wav_path") or "")
        source_stream_path = existing_path(row.get("source_stream_audio_path") or "")
        source_path = existing_path(row.get("source_audio_path") or "")
        run_dir = eval_dir / run_id
        if generated_path is not None:
            run_dir = generated_path.parent
        row["eval_label"] = eval_label
        row["eval_dir"] = str(eval_dir)
        row["compare_backend"] = infer_backend(eval_label, row)
        row["compare_chunk_ms"] = infer_chunk_ms(eval_label, row)
        row["run_dir"] = str(run_dir)
        row["source_audio_path"] = str(source_path or run_dir / "source_eval.wav")
        row["source_stream_audio_path"] = str(source_stream_path or run_dir / "source_stream_24k.wav")
        row["generated_audio_path"] = str(generated_path or run_dir / "generated_target.wav")
        row["run_page_path"] = str(run_dir / "index.html")
    return rows


def existing_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    return path if path.exists() else None


def keep_row(row: dict[str, Any], prefix: str, pattern: re.Pattern[str] | None) -> bool:
    run_id = str(row.get("run_id") or "")
    if prefix and not run_id.startswith(prefix):
        return False
    if pattern and not pattern.search(run_id):
        return False
    return True


def num(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def link(path: str | Path, base: Path, label: str) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    return f'<a href="{esc(href(path, base))}">{esc(label)}</a>'


def audio(path: str | Path, base: Path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    return f'<audio controls preload="metadata" src="{esc(href(path, base))}"></audio>'


def href(path: Path, base: Path) -> str:
    relative = rel(path, base)
    if not relative.startswith("/"):
        return relative
    return path.resolve().as_uri()


def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("run_id") or ""),
        float(row.get("speed_factor") or 0.0),
        int(row.get("compare_chunk_ms") or 0),
        str(row.get("compare_backend") or ""),
    )


def render_rows(rows: list[dict[str, Any]], out_dir: Path) -> str:
    pieces = []
    for row in sorted(rows, key=sort_key):
        chunk_ms = row.get("compare_chunk_ms")
        chunk_s = "" if chunk_ms is None else f"{chunk_ms / 1000.0:.2f}s"
        pieces.append(
            "<tr>"
            f"<td>{esc(row.get('run_id'))}</td>"
            f"<td>{esc(row.get('compare_backend'))}</td>"
            f"<td>{esc(chunk_s)}</td>"
            f"<td>{num(row.get('speed_factor'), 2)}</td>"
            f"<td>{num(row.get('source_stream_duration_s'), 2)}</td>"
            f"<td>{num(row.get('generated_duration_s'), 2)}</td>"
            f"<td>{num(row.get('full_s2s_rtf'), 3)}</td>"
            f"<td>{num(row.get('end_lag_s'), 2)}</td>"
            f"<td>{num(row.get('wall_clock_end_delay_s'), 2)}</td>"
            f"<td>{num(row.get('max_backlog_s'), 2)}</td>"
            f"<td>{num(row.get('max_playback_queue_s'), 2)}</td>"
            f"<td>{num(row.get('xcomet_qe_score'), 4)}</td>"
            f"<td>{num(row.get('metricx_qe_score'), 3)}</td>"
            f"<td>{num(row.get('metricx_qe_error'), 3)}</td>"
            f"<td>{num(row.get('bleu'), 2)}</td>"
            f"<td>{num(row.get('chrf'), 2)}</td>"
            f"<td>{num(row.get('cer'), 3)}</td>"
            f"<td>{link(row.get('run_page_path') or '', out_dir, 'open')}</td>"
            f"<td>{audio(row.get('source_stream_audio_path') or row.get('source_audio_path') or '', out_dir)}</td>"
            f"<td>{audio(row.get('generated_audio_path') or '', out_dir)}</td>"
            "</tr>"
        )
    return "\n".join(pieces)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(args.run_id_regex) if args.run_id_regex else None
    rows: list[dict[str, Any]] = []
    for raw in args.eval:
        label, eval_dir = parse_eval_arg(raw)
        rows.extend(
            row
            for row in load_metrics(label, eval_dir)
            if keep_row(row, args.run_id_prefix, pattern)
        )
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

    (out_dir / "compare_metrics.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in sorted(rows, key=sort_key)) + "\n",
        encoding="utf-8",
    )
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>{esc(args.title)}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;color:#202428}}
h1{{font-size:22px}}table{{border-collapse:collapse;width:100%}}
td,th{{border-top:1px solid #d6d9de;padding:8px;text-align:left;font-size:13px;vertical-align:top}}
audio{{width:220px}}.meta{{color:#667085;margin:8px 0 16px}}
.audio-tools{{display:grid;grid-template-columns:1fr auto;gap:6px;align-items:center;margin-top:4px;color:#667085;font-size:12px}}
.audio-tools input{{width:100%}}
</style>
<h1>{esc(args.title)}</h1>
<div class="meta">
{len(rows)} runs · source audio is the selected source clip · target audio is backend output
<br>duration lag = target audio duration - source stream duration · wall delay = simulated target playback end wall-clock - source stream end · max backlog = max window-level emitted-target deficit
<br>QE columns are reference-free source+hypothesis scores. Current full-run QE uses manifest sentence slots when available; system hypotheses are monotonic text splits into those slots, not manually sentence-aligned. xCOMET-QE is the raw xCOMET-lite no-reference score without artificial rescaling and should be treated as a relative diagnostic when segment scores include negatives. MetricX-QE is higher-is-better, and MetricX err is lower-is-better.
</div>
<table>
<thead>
<tr>
<th>run</th><th>backend</th><th>chunk</th><th>speed</th><th>stream s</th>
<th>target s</th><th>RTF</th><th>duration lag</th><th>wall delay</th><th>max backlog</th><th>max queue</th>
<th>xCOMET-QE raw ↑</th><th>MetricX-QE ↑</th><th>MetricX err ↓</th>
<th>BLEU</th><th>chrF</th><th>CER</th><th>detail</th><th>streamed source</th><th>target</th>
</tr>
</thead>
<tbody>
{render_rows(rows, out_dir)}
</tbody>
</table>
<script>
function fmtTime(s) {{
  if (!Number.isFinite(s)) return "0:00";
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60).toString().padStart(2, "0");
  return `${{m}}:${{r}}`;
}}
document.querySelectorAll("audio").forEach((audio) => {{
  const tools = document.createElement("div");
  tools.className = "audio-tools";
  const range = document.createElement("input");
  range.type = "range";
  range.min = "0";
  range.max = "1";
  range.step = "0.001";
  range.value = "0";
  const time = document.createElement("span");
  time.textContent = "0:00 / 0:00";
  tools.append(range, time);
  audio.insertAdjacentElement("afterend", tools);
  function update() {{
    const duration = Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : 0;
    range.value = duration ? String(audio.currentTime / duration) : "0";
    time.textContent = `${{fmtTime(audio.currentTime)}} / ${{fmtTime(duration)}}`;
  }}
  range.addEventListener("input", () => {{
    if (Number.isFinite(audio.duration) && audio.duration > 0) {{
      audio.currentTime = Number(range.value) * audio.duration;
    }}
  }});
  audio.addEventListener("loadedmetadata", update);
  audio.addEventListener("timeupdate", update);
  update();
}});
</script>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps({"runs": len(rows), "output": str(out_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
