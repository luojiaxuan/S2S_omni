#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.codec_data import align_wav_to_num_frames, load_mono_wav
from s2s_omni.speech_signals import coverage_signals

OmniCodecCollator = None
OmniCodecPairDataset = None
Wav2OmniCodecModel = None
codec_accuracy = None
wav2codec_loss = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate wav -> Omni codec checkpoint.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True, help="final/ model dir or run output dir.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--hop-length", type=int, default=1920)
    parser.add_argument("--num-quantizers", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--audio-samples", type=int, default=8)
    parser.add_argument("--decode-model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--decode-device-map", default="auto")
    parser.add_argument("--asr-model", default="")
    parser.add_argument("--asr-device", default="cuda:0")
    return parser.parse_args()


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    candidates = [
        path,
        path / "final",
        path / "model",
    ]
    if (path / "latest_checkpoint.txt").exists():
        latest = Path((path / "latest_checkpoint.txt").read_text(encoding="utf-8").strip())
        candidates.extend([latest / "model", latest])
    for candidate in candidates:
        if (candidate / "config.json").exists() and (candidate / "model.pt").exists():
            return candidate
    raise FileNotFoundError(f"could not find config.json/model.pt under {path}")


def torch_dtype(name: str):
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    out: dict[str, Any] = {"batches": len(rows)}
    for key in ["loss", "accuracy", "top5_accuracy", "frame_accuracy"]:
        values = [row[key] for row in rows if row.get(key) is not None]
        out[key] = round(sum(values) / len(values), 6) if values else None
    out["valid_codes"] = sum(int(row.get("valid_codes") or 0) for row in rows)
    out["valid_frames"] = sum(int(row.get("valid_frames") or 0) for row in rows)
    per_q: list[list[float]] = [[] for _ in range(16)]
    for row in rows:
        for index, value in enumerate(row.get("per_quantizer_accuracy") or []):
            if value is not None and index < len(per_q):
                per_q[index].append(float(value))
    out["per_quantizer_accuracy"] = [
        round(sum(values) / len(values), 6) if values else None for values in per_q
    ]
    return out


def evaluate_codes(model: Any, loader: Any, device: str, dtype_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    dtype = torch_dtype(dtype_name)
    model.to(device)
    model.eval()
    rows = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            wav = batch["wav"].to(device)
            frame_mask = batch["frame_mask"].to(device)
            codes = batch["codes"].to(device)
            with torch.autocast(
                device_type="cuda" if str(device).startswith("cuda") else "cpu",
                dtype=dtype,
                enabled=dtype_name != "float32",
            ):
                logits = model(wav, frame_mask)
                loss = wav2codec_loss(logits, codes)
            metrics = codec_accuracy(logits.float(), codes, topk=(1, 5))
            metrics["loss"] = round(float(loss.detach().cpu().item()), 6)
            metrics["batch_index"] = batch_index
            rows.append(metrics)
    return aggregate_metrics(rows), rows


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


def mel_l1(left: np.ndarray, right: np.ndarray, sample_rate: int) -> float | None:
    try:
        import librosa
    except ModuleNotFoundError:
        return None
    n = min(left.shape[0], right.shape[0])
    if n <= 0:
        return None
    left_mel = librosa.feature.melspectrogram(y=left[:n], sr=sample_rate, n_mels=80)
    right_mel = librosa.feature.melspectrogram(y=right[:n], sr=sample_rate, n_mels=80)
    left_db = librosa.power_to_db(left_mel, ref=np.max)
    right_db = librosa.power_to_db(right_mel, ref=np.max)
    return round(float(np.mean(np.abs(left_db - right_db))), 6)


class OmniCode2WavDecoder:
    def __init__(self, model_name: str, device_map: str) -> None:
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration

        self.torch = torch
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype="auto",
            device_map=device_map,
        )
        self.model.eval()
        if not getattr(self.model, "has_talker", False):
            self.model.enable_talker()
        self.device = self.model.code2wav.device

    def decode(self, codes: Any) -> np.ndarray:
        if codes.ndim == 2:
            codes = codes.unsqueeze(0)
        codes = codes.to(self.device)
        with self.torch.inference_mode():
            wav = self.model.code2wav.chunked_decode(codes.long(), chunk_size=300, left_context_size=25)
        return wav.reshape(-1).detach().float().cpu().numpy()


def make_asr_pipe(model_name: str, device: str):
    if not model_name:
        return None
    import torch
    from transformers import pipeline

    torch_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        torch_dtype=torch_dtype,
        device=device,
    )


def transcribe(pipe: Any, wav_path: Path) -> str:
    if pipe is None:
        return ""
    result = pipe(
        str(wav_path),
        generate_kwargs={"language": "zh", "task": "transcribe"},
        return_timestamps=False,
    )
    if isinstance(result, dict):
        return str(result.get("text") or "").strip()
    return str(result).strip()


def render_html(rows: list[dict[str, Any]], output: Path, title: str) -> None:
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
        sections.append(
            f"""
  <section>
    <h2>{html.escape(str(row.get("id") or ""))}</h2>
    <p><strong>Source:</strong> {html.escape(str(row.get("source_text") or ""))}</p>
    <p><strong>Generated text:</strong> {html.escape(str(row.get("generated_text") or ""))}</p>
    <div class="grid">
      <div><h3>Gold wav</h3><audio controls src="{html.escape(rel(row.get("gold_wav")))}"></audio></div>
      <div><h3>Gold codes decode</h3><audio controls src="{html.escape(rel(row.get("gold_decode_wav")))}"></audio></div>
      <div><h3>Pred codes decode</h3><audio controls src="{html.escape(rel(row.get("pred_decode_wav")))}"></audio></div>
    </div>
    <pre>{html.escape(json.dumps(row.get("metrics") or {}, ensure_ascii=False, indent=2))}</pre>
  </section>"""
        )
    output.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #1f2328; }}
    section {{ border: 1px solid #d0d7de; border-radius: 8px; padding: 16px; margin: 18px 0; }}
    h1 {{ font-size: 22px; }}
    h2 {{ font-size: 16px; margin-top: 0; }}
    h3 {{ font-size: 14px; margin-bottom: 6px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
    audio {{ width: 100%; }}
    pre {{ white-space: pre-wrap; background: #f6f8fa; padding: 10px; border-radius: 6px; }}
    @media (max-width: 820px) {{ .grid {{ grid-template-columns: 1fr; }} body {{ margin: 18px; }} }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
{"".join(sections)}
</body>
</html>
""",
        encoding="utf-8",
    )


def make_audio_samples(args: argparse.Namespace, model: Any, dataset: OmniCodecPairDataset, output_dir: Path) -> list[dict[str, Any]]:
    import torch

    sample_count = min(args.audio_samples, len(dataset))
    if sample_count <= 0:
        return []
    decoder = OmniCode2WavDecoder(args.decode_model, args.decode_device_map)
    asr_pipe = make_asr_pipe(args.asr_model, args.asr_device)
    audio_dir = output_dir / "audio"
    rows: list[dict[str, Any]] = []
    model.eval()
    model.to(args.device)
    for index in range(sample_count):
        item = dataset[index]
        record = item["record"]
        wav = align_wav_to_num_frames(item["wav"], item["num_frames"], args.hop_length)
        codes = torch.from_numpy(item["codes"]).long()
        batch_wav = torch.from_numpy(wav).unsqueeze(0).to(args.device)
        frame_mask = torch.ones((1, item["num_frames"]), dtype=torch.bool, device=args.device)
        with torch.no_grad():
            logits = model(batch_wav, frame_mask).float().cpu()
        pred_codes = logits.argmax(dim=-1)[0].long()
        gold_decoded = decoder.decode(codes)
        pred_decoded = decoder.decode(pred_codes)

        safe_id = re_safe(str(item["id"]))
        gold_wav_path = audio_dir / f"{safe_id}__gold.wav"
        gold_decode_path = audio_dir / f"{safe_id}__gold_codes.wav"
        pred_decode_path = audio_dir / f"{safe_id}__pred_codes.wav"
        write_wav(gold_wav_path, wav, args.sample_rate)
        gold_duration = write_wav(gold_decode_path, gold_decoded, args.sample_rate)
        pred_duration = write_wav(pred_decode_path, pred_decoded, args.sample_rate)
        asr_text = transcribe(asr_pipe, pred_decode_path)
        coverage = (
            coverage_signals(str(record.get("generated_text") or ""), asr_text, "zh")
            if asr_text
            else None
        )
        metrics = {
            "gold_duration_s": gold_duration,
            "pred_duration_s": pred_duration,
            "duration_delta_s": round(abs(pred_duration - gold_duration), 6),
            "wav_l1_vs_gold_code_decode": wav_l1(pred_decoded, gold_decoded),
            "mel_l1_vs_gold_code_decode": mel_l1(pred_decoded, gold_decoded, args.sample_rate),
            "pred_asr_text": asr_text,
            "pred_asr_coverage": coverage,
        }
        rows.append(
            {
                "id": item["id"],
                "source_text": record.get("source_text"),
                "generated_text": record.get("generated_text"),
                "gold_wav": str(gold_wav_path),
                "gold_decode_wav": str(gold_decode_path),
                "pred_decode_wav": str(pred_decode_path),
                "metrics": metrics,
            }
        )
    return rows


def re_safe(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180] or "sample"


def main() -> None:
    args = parse_args()
    global OmniCodecCollator
    global OmniCodecPairDataset
    global Wav2OmniCodecModel
    global codec_accuracy
    global wav2codec_loss
    from s2s_omni.wav2codec import (
        OmniCodecCollator,
        OmniCodecPairDataset,
        Wav2OmniCodecModel,
        codec_accuracy,
        wav2codec_loss,
    )

    import torch
    from torch.utils.data import DataLoader

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_checkpoint(args.checkpoint)
    model = Wav2OmniCodecModel.from_pretrained(checkpoint, map_location="cpu")
    dataset = OmniCodecPairDataset(
        args.manifest,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        num_quantizers=args.num_quantizers,
        max_frames=args.max_frames,
        random_crop=False,
        max_records=args.max_records,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=OmniCodecCollator(hop_length=args.hop_length),
    )
    summary, batch_metrics = evaluate_codes(model, loader, args.device, args.dtype)
    (output_dir / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "eval_batches.json").write_text(
        json.dumps(batch_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    sample_rows = []
    if args.audio_samples > 0:
        sample_rows = make_audio_samples(args, model, dataset, output_dir)
        (output_dir / "audio_samples.json").write_text(
            json.dumps(sample_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        render_html(sample_rows, output_dir / "index.html", "wav2omni-codec audio eval")

    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "manifest": args.manifest,
                "output_dir": str(output_dir),
                "summary": summary,
                "audio_samples": len(sample_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
