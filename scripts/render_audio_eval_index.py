#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an HTML page for audio eval samples.")
    parser.add_argument("--manifest", required=True, help="manifest.json or manifest.jsonl from audio eval.")
    parser.add_argument("--output", default=None, help="Output HTML path. Defaults to index.html next to manifest.")
    parser.add_argument("--title", default="S2S Omni Audio Eval")
    return parser.parse_args()


def load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {}, rows
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("samples", [])
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a samples list")
    return data, rows


def rel_audio(path: str | None, base_dir: Path) -> str:
    if not path:
        return ""
    audio_path = Path(path)
    if audio_path.is_absolute():
        try:
            return audio_path.relative_to(base_dir).as_posix()
        except ValueError:
            return audio_path.as_posix()
    return audio_path.as_posix()


def text_len(text: str | None) -> int:
    return len((text or "").strip())


def metric(row: dict[str, Any], prefix: str, key: str) -> Any:
    return row.get(f"{prefix}_{key}")


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_panel(row: dict[str, Any], prefix: str, base_dir: Path) -> str:
    label = "Base" if prefix == "base" else "Evo"
    text = str(row.get(f"{prefix}_prediction") or "")
    audio = rel_audio(row.get(f"{prefix}_audio"), base_dir)
    duration = row.get(f"{prefix}_duration_s")
    budget_ratio = metric(row, prefix, "target_budget_ratio")
    rtf = metric(row, prefix, "s2s_rtf")
    bag_f1 = metric(row, prefix, "bag_f1_vs_reference")
    audio_tag = (
        f'<audio controls src="{html.escape(audio)}"></audio>'
        if audio
        else '<div class="missing">No audio.</div>'
    )
    return f"""
      <div class="panel">
        <div class="label">{label}</div>
        <div class="meta">{text_len(text)} chars, {fmt(duration)}s, budget {fmt(budget_ratio)}, RTF {fmt(rtf)}, bag-F1 {fmt(bag_f1)}</div>
        {audio_tag}
        <p>{html.escape(text)}</p>
      </div>"""


def render(manifest: dict[str, Any], rows: list[dict[str, Any]], output: Path, title: str) -> str:
    base_dir = output.parent.resolve()
    model = manifest.get("model")
    adapter = manifest.get("adapter")
    speaker = manifest.get("speaker")
    note_items = [
        "Qwen3-Omni audio samples",
        f"model: {model}" if model else None,
        f"adapter: {adapter}" if adapter else None,
        f"speaker: {speaker}" if speaker else None,
    ]
    note = " | ".join(item for item in note_items if item)
    sections = []
    for row in rows:
        source = row.get("source_text") or ""
        reference = row.get("reference_translation") or ""
        sections.append(
            f"""
  <section class="sample">
    <div class="id">{html.escape(str(row.get("id") or ""))}</div>
    <div class="source"><strong>Source:</strong> {html.escape(str(source))}</div>
    <div class="source"><strong>Reference:</strong> {html.escape(str(reference))}</div>
    <div class="grid">
{render_panel(row, "base", base_dir)}
{render_panel(row, "evo", base_dir)}
    </div>
  </section>"""
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #1f2328; line-height: 1.45; }}
    h1 {{ font-size: 22px; margin: 0 0 8px; }}
    .note {{ color: #57606a; margin-bottom: 22px; overflow-wrap: anywhere; }}
    .sample {{ border: 1px solid #d0d7de; border-radius: 8px; padding: 16px; margin: 18px 0; }}
    .id {{ font-weight: 650; margin-bottom: 8px; }}
    .source {{ color: #424a53; font-size: 14px; margin: 6px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px; }}
    .panel {{ background: #f6f8fa; border-radius: 8px; padding: 12px; }}
    .label {{ font-weight: 650; margin-bottom: 6px; }}
    .meta {{ color: #57606a; font-size: 13px; margin-bottom: 8px; }}
    .missing {{ color: #8c2f0b; font-size: 13px; margin: 8px 0; }}
    audio {{ width: 100%; margin: 6px 0 10px; }}
    p {{ margin: 0; overflow-wrap: anywhere; }}
    @media (max-width: 760px) {{ .grid {{ grid-template-columns: 1fr; }} body {{ margin: 18px; }} }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="note">{html.escape(note)}</div>
{"".join(sections)}
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output = Path(args.output) if args.output else manifest_path.parent / "index.html"
    manifest, rows = load_manifest(manifest_path)
    output.write_text(render(manifest, rows, output, args.title), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
