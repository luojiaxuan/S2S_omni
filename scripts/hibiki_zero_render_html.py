#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Hibiki-Zero S2S listening samples.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Hibiki-Zero Backlog-Aware S2S Samples")
    parser.add_argument("--max-samples", type=int, default=50)
    return parser.parse_args()


def keyed_predictions(path: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    return {str(row.get("sample_id") or row.get("id")): row for row in read_jsonl(path)}


def esc(value: Any) -> str:
    return html.escape(str(value or ""))


def rel_path(path: str | None, base_dir: Path) -> str:
    if not path:
        return ""
    value = Path(str(path))
    if value.is_absolute():
        try:
            return value.relative_to(base_dir).as_posix()
        except ValueError:
            return value.as_posix()
    return value.as_posix()


def audio(path: str | None, label: str, base_dir: Path) -> str:
    if not path:
        return f'<div class="missing">{esc(label)}: missing</div>'
    return f'<div><span>{esc(label)}</span><audio controls src="{esc(rel_path(path, base_dir))}"></audio></div>'


def render_sample(row: dict[str, Any], pred: dict[str, Any] | None, base_dir: Path) -> str:
    chunks = row.get("source_audio_chunks") or []
    target_wavs = row.get("target_chunk_wavs") or []
    rtf = row.get("speech_s2s_rtf") or []
    chunk_blocks = []
    for idx, chunk in enumerate(chunks):
        pred_wav = ""
        if pred:
            pred_wavs = pred.get("generated_chunk_wavs") or pred.get("target_chunk_wavs") or []
            pred_wav = pred_wavs[idx] if idx < len(pred_wavs) else pred.get("generated_wav_path", "")
        chunk_blocks.append(
            f"""
      <div class="chunk">
        <div class="chunk-title">chunk {idx} · RTF {esc(rtf[idx] if idx < len(rtf) else '')}</div>
        <p><b>target text:</b> {esc(chunk.get('compressed_en_text'))}</p>
        {audio(chunk.get('source_audio'), 'source', base_dir)}
        {audio(target_wavs[idx] if idx < len(target_wavs) else '', 'teacher target', base_dir)}
        {audio(pred_wav, 'model output', base_dir) if pred else ''}
      </div>"""
        )
    return f"""
  <section>
    <h2>{esc(row.get('sample_id'))}</h2>
    <p><b>lang:</b> {esc(row.get('src_lang'))} -> en · <b>speed:</b> {esc(row.get('speed_factor'))}</p>
    <p><b>compressed:</b> {esc(row.get('compressed_en_text'))}</p>
    <p><b>reference:</b> {esc(row.get('reference_en_text'))}</p>
    {audio(row.get('target_en_wav'), 'full teacher target', base_dir)}
    <div class="chunks">{''.join(chunk_blocks)}</div>
  </section>"""


def render(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    title: str,
    base_dir: Path,
) -> str:
    sections = []
    for row in rows:
        key = str(row.get("sample_id") or row.get("id"))
        sections.append(render_sample(row, predictions.get(key), base_dir))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #24292f; line-height: 1.45; }}
    h1 {{ font-size: 22px; margin: 0 0 16px; }}
    h2 {{ font-size: 16px; margin: 0 0 8px; }}
    section {{ border-top: 1px solid #d0d7de; padding: 18px 0; }}
    .chunks {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 12px; }}
    .chunk {{ border: 1px solid #d8dee4; border-radius: 8px; padding: 12px; background: #f6f8fa; }}
    .chunk-title {{ font-weight: 650; color: #57606a; font-size: 13px; }}
    audio {{ display: block; width: 100%; margin: 4px 0 10px; }}
    span {{ font-size: 12px; color: #57606a; }}
    p {{ overflow-wrap: anywhere; }}
    .missing {{ color: #8c2f0b; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>{esc(title)}</h1>
  {''.join(sections)}
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.manifest)[: args.max_samples]
    predictions = keyed_predictions(args.predictions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(rows, predictions, args.title, output.parent.resolve()), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
