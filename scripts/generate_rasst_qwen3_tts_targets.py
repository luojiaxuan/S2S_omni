#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.audio import load_audio_span, resample_audio
from s2s_omni.rasst import (
    DEFAULT_RASST_DEV,
    DEFAULT_RASST_TRAIN,
    iter_rasst_rows,
    sanitize_id,
    write_mono_wav,
)
from s2s_omni.textgrid import transcript_for_mfa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full-sentence Qwen3-TTS target wavs for plain RASST baseline rows."
    )
    parser.add_argument("--input", default=DEFAULT_RASST_TRAIN)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--ref-chunks", type=int, default=3)
    parser.add_argument("--ref-sample-rate", type=int, default=16000)
    parser.add_argument("--max-ref-seconds", type=float, default=12.0)
    parser.add_argument("--x-vector-only-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ref-text", default="")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-rejected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--write-mfa-corpus", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dev", action="store_true", help=f"Shortcut for --input {DEFAULT_RASST_DEV}")
    return parser.parse_args()


def torch_dtype(name: str) -> Any:
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def apply_optional_qwen_tts_compat() -> None:
    try:
        from sglang_omni.models.qwen3_tts.compat import (
            apply_qwen_tts_transformers_compatibility_patches,
        )
    except Exception:
        return
    apply_qwen_tts_transformers_compatibility_patches()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")
        handle.flush()


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("row_id"):
                out.add(str(row["row_id"]))
    return out


def audio_to_numpy(wav: Any) -> np.ndarray:
    if hasattr(wav, "detach"):
        wav = wav.reshape(-1).detach().float().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    return np.clip(wav, -1.0, 1.0)


def reference_audio_for_row(row: Any, args: argparse.Namespace) -> tuple[np.ndarray, int]:
    chunks: list[np.ndarray] = []
    max_samples = int(max(1.0, args.max_ref_seconds) * args.ref_sample_rate)
    total = 0
    for turn in row.turns[: max(1, args.ref_chunks)]:
        wav, sr = load_audio_span(turn.audio_path, target_sample_rate=args.ref_sample_rate)
        if wav.size == 0:
            continue
        remaining = max_samples - total
        if remaining <= 0:
            break
        wav = wav[:remaining]
        chunks.append(wav.astype(np.float32, copy=False))
        total += int(wav.size)
        if total >= max_samples:
            break
    if not chunks:
        raise RuntimeError("failed to load non-empty source reference audio")
    return np.concatenate(chunks), int(args.ref_sample_rate)


class Qwen3TtsTargetGenerator:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch

        sys.modules.setdefault("kernels", None)
        apply_optional_qwen_tts_compat()
        from qwen_tts import Qwen3TTSModel

        self.args = args
        self.torch = torch
        started = time.perf_counter()
        self.model = Qwen3TTSModel.from_pretrained(
            args.model,
            device_map=args.device_map,
            dtype=torch_dtype(args.dtype),
            attn_implementation=args.attn_implementation,
        )
        self.load_s = round(time.perf_counter() - started, 6)

    def generate_one(self, row: Any) -> tuple[np.ndarray, int, float]:
        ref_audio = reference_audio_for_row(row, self.args)
        target_text = row.full_target_text
        started = time.perf_counter()
        with self.torch.inference_mode():
            if self.args.x_vector_only_mode:
                prompt = self.model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=self.args.ref_text,
                    x_vector_only_mode=True,
                )
                wavs, sr = self.model.generate_voice_clone(
                    text=target_text,
                    language=self.args.language,
                    voice_clone_prompt=prompt,
                    max_new_tokens=self.args.max_new_tokens,
                )
            else:
                wavs, sr = self.model.generate_voice_clone(
                    text=target_text,
                    language=self.args.language,
                    ref_audio=ref_audio,
                    ref_text=self.args.ref_text,
                    max_new_tokens=self.args.max_new_tokens,
                )
        wav = audio_to_numpy(wavs[0])
        sr = int(sr)
        if sr != int(self.args.sample_rate):
            wav = resample_audio(wav, sr, int(self.args.sample_rate))
            sr = int(self.args.sample_rate)
        return wav, sr, round(time.perf_counter() - started, 6)


def main() -> None:
    args = parse_args()
    if args.dev:
        args.input = DEFAULT_RASST_DEV
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    if not args.x_vector_only_mode and not args.ref_text:
        raise SystemExit("--ref-text is required unless --x-vector-only-mode is enabled")

    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "target_wav"
    mfa_dir = output_dir / "mfa_corpus"
    accepted_path = output_dir / "target_manifest.jsonl"
    rejected_path = output_dir / "target_rejected.jsonl"
    if not args.resume:
        for path in [accepted_path, rejected_path]:
            if path.exists():
                path.unlink()
    done = existing_ids(accepted_path) | existing_ids(rejected_path)

    selected = []
    eligible_index = 0
    for row in iter_rasst_rows(args.input, max_records=args.max_records):
        if row.row_id in done:
            continue
        if not row.full_target_text:
            append_jsonl(
                rejected_path,
                {
                    "row_id": row.row_id,
                    "accepted": False,
                    "reject_reasons": ["empty_full_target_text"],
                    "source_path": row.source_path,
                },
            )
            continue
        if eligible_index % args.num_shards == args.shard_index:
            selected.append(row)
        eligible_index += 1

    if not selected:
        print(json.dumps({"selected": 0, "done": len(done)}, ensure_ascii=False, indent=2))
        return

    generator = Qwen3TtsTargetGenerator(args)
    accepted = 0
    rejected = 0
    for index, row in enumerate(selected, start=1):
        safe_id = sanitize_id(row.row_id)
        out_record: dict[str, Any] = {
            "row_id": row.row_id,
            "source_path": row.source_path,
            "row_index": row.index,
            "full_target_text": row.full_target_text,
            "target_char_spans": [list(span) for span in row.target_char_spans],
            "chunks": [
                {
                    "chunk_index": turn.index,
                    "target_text": turn.assistant_text,
                    "source_audio": turn.audio_path,
                    "char_span": list(row.target_char_spans[turn.index]),
                }
                for turn in row.turns
            ],
            "model": args.model,
            "backend": "qwen3_tts",
            "language": args.language,
            "sample_rate": args.sample_rate,
            "x_vector_only_mode": args.x_vector_only_mode,
            "ref_chunks": args.ref_chunks,
            "max_ref_seconds": args.max_ref_seconds,
            "model_load_s": generator.load_s,
        }
        try:
            wav, sr, synth_s = generator.generate_one(row)
            if wav.size == 0:
                raise RuntimeError("empty generated wav")
            wav_path = wav_dir / f"{safe_id}.wav"
            write_mono_wav(wav_path, wav, sr)
            out_record.update(
                {
                    "accepted": True,
                    "target_wav_path": str(wav_path),
                    "target_duration_s": round(float(wav.size) / float(sr), 6),
                    "target_codes_path": None,
                    "codec_frames": None,
                    "codec_num_quantizers": None,
                    "synth_s": synth_s,
                    "reject_reasons": [],
                }
            )
            if args.write_mfa_corpus:
                mfa_wav = mfa_dir / f"{safe_id}.wav"
                mfa_txt = mfa_dir / f"{safe_id}.txt"
                write_mono_wav(mfa_wav, wav, sr)
                mfa_txt.parent.mkdir(parents=True, exist_ok=True)
                mfa_txt.write_text(transcript_for_mfa(row.full_target_text), encoding="utf-8")
                out_record["mfa_wav_path"] = str(mfa_wav)
                out_record["mfa_txt_path"] = str(mfa_txt)
            append_jsonl(accepted_path, out_record)
            accepted += 1
        except Exception as exc:
            out_record.update(
                {
                    "accepted": False,
                    "reject_reasons": [f"exception:{type(exc).__name__}"],
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            rejected += 1
            if args.save_rejected:
                append_jsonl(rejected_path, out_record)
        if args.log_every > 0 and index % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "processed": index,
                        "accepted": accepted,
                        "rejected": rejected,
                        "last_row_id": row.row_id,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = {
        "input": args.input,
        "output_dir": str(output_dir),
        "selected_this_run": len(selected),
        "accepted_this_run": accepted,
        "rejected_this_run": rejected,
        "accepted_total": len(existing_ids(accepted_path)),
        "rejected_total": len(existing_ids(rejected_path)),
        "model": args.model,
        "backend": "qwen3_tts",
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "x_vector_only_mode": args.x_vector_only_mode,
    }
    (output_dir / f"summary_shard{args.shard_index}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
