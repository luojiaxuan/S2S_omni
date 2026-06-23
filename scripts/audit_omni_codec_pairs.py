#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.codec_data import load_mono_wav, resolve_manifest_path
from s2s_omni.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Omni self-domain wav/code pair manifests.")
    parser.add_argument("--input", required=True, help="pairs.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--hop-length", type=int, default=1920)
    parser.add_argument("--num-quantizers", type=int, default=16)
    parser.add_argument("--codebook-size", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--decode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180] or "sample"


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> float:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    sf.write(path, np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16")
    return round(float(audio.shape[0]) / float(sample_rate), 6)


def wav_l1(left: np.ndarray, right: np.ndarray) -> float | None:
    n = min(left.shape[0], right.shape[0])
    if n <= 0:
        return None
    return round(float(np.mean(np.abs(left[:n] - right[:n]))), 6)


class Code2WavDecoder:
    def __init__(self, model_name: str, device_map: str) -> None:
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration

        self.torch = torch
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype="auto",
            device_map=device_map,
        ).eval()
        if not getattr(self.model, "has_talker", False):
            self.model.enable_talker()
        self.device = self.model.code2wav.device

    def decode(self, codes: np.ndarray) -> np.ndarray:
        tensor = self.torch.from_numpy(codes.astype(np.int64, copy=False)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            wav = self.model.code2wav.chunked_decode(tensor, chunk_size=300, left_context_size=25)
        return wav.reshape(-1).detach().float().cpu().numpy()


def audit_record(
    record: dict[str, Any],
    manifest_path: Path,
    args: argparse.Namespace,
    decoder: Code2WavDecoder | None,
    audio_dir: Path,
) -> dict[str, Any]:
    codes_path = resolve_manifest_path(record["codes_path"], manifest_path)
    wav_path = resolve_manifest_path(record["wav_path"], manifest_path)
    codes = np.load(codes_path)
    if codes.ndim == 3 and codes.shape[0] == 1:
        codes = codes[0]
    wav = load_mono_wav(wav_path, args.sample_rate)
    expected_samples = int(codes.shape[1]) * int(args.hop_length) if codes.ndim == 2 else 0
    reasons = []
    if codes.ndim != 2:
        reasons.append(f"bad_code_rank:{codes.shape}")
    elif codes.shape[0] != args.num_quantizers:
        reasons.append(f"bad_num_quantizers:{codes.shape}")
    if codes.size:
        code_min = int(codes.min())
        code_max = int(codes.max())
        if code_min < 0 or code_max >= args.codebook_size:
            reasons.append(f"code_out_of_range:{code_min}:{code_max}")
    else:
        code_min = None
        code_max = None
        reasons.append("empty_codes")
    if expected_samples and abs(wav.shape[0] - expected_samples) > args.hop_length:
        reasons.append("wav_code_length_mismatch")
    decoded_path = None
    decoded_duration_s = None
    decoded_l1 = None
    if decoder is not None and not reasons:
        decoded = decoder.decode(codes)
        decoded_path = audio_dir / f"{safe_name(str(record.get('id')))}__gold_code_decode.wav"
        decoded_duration_s = write_wav(decoded_path, decoded, args.sample_rate)
        decoded_l1 = wav_l1(decoded, wav)
    return {
        "id": record.get("id"),
        "base_id": record.get("base_id"),
        "generated_text": record.get("generated_text"),
        "wav_path": str(wav_path),
        "codes_path": str(codes_path),
        "decoded_wav_path": str(decoded_path) if decoded_path else None,
        "code_shape": [int(v) for v in codes.shape],
        "code_min": code_min,
        "code_max": code_max,
        "wav_samples": int(wav.shape[0]),
        "expected_samples": expected_samples,
        "duration_s": round(float(wav.shape[0]) / float(args.sample_rate), 6),
        "decoded_duration_s": decoded_duration_s,
        "decoded_l1_vs_saved_wav": decoded_l1,
        "gate_pass": not reasons,
        "gate_reasons": reasons,
        "source_text": record.get("source_text"),
        "reference_translation": record.get("reference_translation"),
    }


def render_html(rows: list[dict[str, Any]], output: Path) -> None:
    def rel(path: str | None) -> str:
        if not path:
            return ""
        p = Path(path)
        try:
            return p.relative_to(output.parent).as_posix()
        except ValueError:
            return p.as_posix()

    sections = []
    for row in rows:
        decoded = (
            f'<div><h3>Code2wav decode</h3><audio controls src="{html.escape(rel(row.get("decoded_wav_path")))}"></audio></div>'
            if row.get("decoded_wav_path")
            else ""
        )
        sections.append(
            f"""
  <section>
    <h2>{html.escape(str(row.get("id") or ""))}</h2>
    <p><strong>Source:</strong> {html.escape(str(row.get("source_text") or ""))}</p>
    <p><strong>Generated:</strong> {html.escape(str(row.get("generated_text") or ""))}</p>
    <div class="grid">
      <div><h3>Saved wav</h3><audio controls src="{html.escape(rel(row.get("wav_path")))}"></audio></div>
      {decoded}
    </div>
    <pre>{html.escape(json.dumps(row, ensure_ascii=False, indent=2))}</pre>
  </section>"""
        )
    output.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Omni codec pair audit</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #1f2328; }}
    section {{ border: 1px solid #d0d7de; border-radius: 8px; padding: 16px; margin: 18px 0; }}
    h1 {{ font-size: 22px; }}
    h2 {{ font-size: 16px; margin-top: 0; }}
    h3 {{ font-size: 14px; margin-bottom: 6px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }}
    audio {{ width: 100%; }}
    pre {{ white-space: pre-wrap; background: #f6f8fa; padding: 10px; border-radius: 6px; }}
    @media (max-width: 760px) {{ .grid {{ grid-template-columns: 1fr; }} body {{ margin: 18px; }} }}
  </style>
</head>
<body>
  <h1>Omni codec pair audit</h1>
{"".join(sections)}
</body>
</html>
""",
        encoding="utf-8",
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for row in rows if row.get("gate_pass"))
    durations = [float(row["duration_s"]) for row in rows if row.get("duration_s") is not None]
    l1_values = [
        float(row["decoded_l1_vs_saved_wav"])
        for row in rows
        if row.get("decoded_l1_vs_saved_wav") is not None
    ]
    return {
        "records": len(rows),
        "gate_passed": passed,
        "gate_failed": len(rows) - passed,
        "duration_s": stats(durations),
        "decoded_l1_vs_saved_wav": stats(l1_values),
    }


def stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    values = sorted(values)
    return {
        "min": round(values[0], 6),
        "p50": round(values[len(values) // 2], 6),
        "p90": round(values[min(len(values) - 1, int(len(values) * 0.9))], 6),
        "max": round(values[-1], 6),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)
    records = read_jsonl(input_path)
    sample_records = records[: args.num_samples] if args.num_samples > 0 else records
    decoder = Code2WavDecoder(args.model, args.device_map) if args.decode else None
    rows = [
        audit_record(record, input_path, args, decoder, output_dir / "audio")
        for record in sample_records
    ]
    write_jsonl(output_dir / "audit_rows.jsonl", rows)
    summary = summarize(rows)
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    render_html(rows, output_dir / "index.html")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
