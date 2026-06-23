#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.io import read_jsonl
from s2s_omni.omni_talker import (
    generate_talker_codes,
    patch_qwen3_omni_no_split,
    prepare_talker_condition,
    soft_code2wav,
)
from s2s_omni.rasst import write_mono_wav
from scripts.generate_rasst_softwav_outputs import merge_lora_into_omni
from scripts.train_qwen3_omni_softwav_lora import load_processor_inputs, wav_only_logits_by_q


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render teacher-forced Qwen3-Omni talker wavs from a RASST soft-wav manifest."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--adapter")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--speaker", default="Ethan")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--render-mode", choices=["generate", "manual-st-argmax"], default="generate")
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--tokens-from-target-duration", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--codec-frame-rate", type=float, default=12.5)
    parser.add_argument("--token-margin", type=int, default=8)
    parser.add_argument("--talker-do-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--talker-temperature", type=float, default=0.9)
    parser.add_argument("--talker-top-k", type=int, default=50)
    parser.add_argument("--talker-top-p", type=float, default=1.0)
    parser.add_argument("--talker-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--suppress-primary-special", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--primary-from-sequences", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def audio_to_numpy(audio: Any) -> np.ndarray:
    if audio is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(audio, "detach"):
        audio = audio.reshape(-1).detach().float().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
        handle.flush()


class TeacherForcedTalkerRenderer:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        self.args = args
        self.torch = torch
        torch.manual_seed(260623)
        patch_qwen3_omni_no_split(Qwen3OmniMoeForConditionalGeneration)
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(args.model, trust_remote_code=True)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            args.model,
            trust_remote_code=True,
            dtype="auto",
            device_map=args.device_map,
        ).eval()
        if not getattr(self.model, "has_talker", False):
            self.model.enable_talker()
        self.merge_stats = None
        if args.adapter:
            self.merge_stats = merge_lora_into_omni(self.model, args.adapter)
            print(json.dumps({"manual_lora_merge": self.merge_stats}, ensure_ascii=False), flush=True)

    def token_budget(self, record: dict[str, Any]) -> tuple[int | None, int]:
        if not self.args.tokens_from_target_duration:
            min_tokens = self.args.min_new_tokens if self.args.min_new_tokens > 0 else None
            return min_tokens, self.args.max_new_tokens
        target_duration = float(record.get("target_duration_s") or 0.0)
        frames = max(1, int(np.ceil(target_duration * float(self.args.codec_frame_rate))))
        min_tokens = max(1, frames)
        max_tokens = max(min_tokens + 1, min(self.args.max_new_tokens, frames + self.args.token_margin))
        return min_tokens, max_tokens

    def render_one(self, record: dict[str, Any]) -> tuple[np.ndarray, int]:
        model = self.model
        with self.torch.inference_mode():
            inputs, _prompt_len, _target_token_len = load_processor_inputs(
                self.processor,
                record,
                model,
            )
            condition = prepare_talker_condition(
                model,
                inputs,
                speaker=self.args.speaker,
                detach_thinker=True,
            )
            if self.args.render_mode == "manual-st-argmax":
                target_duration = float(record.get("target_duration_s") or 0.0)
                frames = max(1, int(np.ceil(target_duration * float(self.args.codec_frame_rate))))
                frames = min(frames, int(self.args.max_new_tokens))
                logits_by_q = wav_only_logits_by_q(
                    model,
                    condition,
                    frames,
                    temperature=1.0,
                    mode="st_argmax",
                    detach_frame_feedback=True,
                )
                wav = soft_code2wav(
                    model.code2wav,
                    logits_by_q,
                    temperature=1.0,
                    mode="st_argmax",
                )
                if hasattr(wav, "detach") and not bool(self.torch.isfinite(wav).all().detach().cpu()):
                    raise RuntimeError("manual-st-argmax produced non-finite wav")
                wav_np = audio_to_numpy(wav)
                return wav_np, frames
            min_new_tokens, max_new_tokens = self.token_budget(record)
            codes = generate_talker_codes(
                model,
                condition,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=self.args.talker_do_sample,
                temperature=self.args.talker_temperature,
                top_k=self.args.talker_top_k,
                top_p=self.args.talker_top_p,
                repetition_penalty=self.args.talker_repetition_penalty,
                suppress_primary_special=self.args.suppress_primary_special,
                primary_from_sequences=self.args.primary_from_sequences,
            )
            wav = model.code2wav.chunked_decode(codes, chunk_size=300, left_context_size=25)
        return audio_to_numpy(wav), int(codes.shape[-1])


def write_silence(path: Path, sample_rate: int) -> None:
    write_mono_wav(path, np.zeros((1,), dtype=np.float32), sample_rate)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "wav"
    pred_path = output_dir / "predictions.jsonl"
    if pred_path.exists():
        pred_path.unlink()
    records = read_jsonl(args.manifest)
    if args.max_records > 0:
        records = records[: args.max_records]
    renderer = TeacherForcedTalkerRenderer(args)
    for index, record in enumerate(records, start=1):
        wav_path = wav_dir / f"{record['id']}.wav"
        row = {
            "id": record.get("id"),
            "target_text": record.get("target_text"),
            "prediction_text": record.get("target_text"),
        }
        if record.get("expected_empty_target") or not str(record.get("target_text") or "").strip():
            write_silence(wav_path, args.sample_rate)
            row.update(
                {
                    "accepted": True,
                    "generated_wav_path": str(wav_path),
                    "generated_duration_s": 0.0,
                    "generated_codec_frames": 0,
                    "expected_empty_target": True,
                }
            )
        else:
            try:
                wav, frames = renderer.render_one(record)
                write_mono_wav(wav_path, wav, args.sample_rate)
                row.update(
                    {
                        "accepted": True,
                        "generated_wav_path": str(wav_path),
                        "generated_duration_s": round(float(wav.size) / float(args.sample_rate), 6),
                        "generated_codec_frames": frames,
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "accepted": False,
                        "error": str(exc),
                        "reject_reasons": [f"exception:{type(exc).__name__}"],
                    }
                )
        append_jsonl(pred_path, row)
        if args.log_every > 0 and index % args.log_every == 0:
            print(json.dumps({"processed": index, "last_id": record.get("id")}, ensure_ascii=False), flush=True)
    print(json.dumps({"output": str(pred_path), "records": len(records)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
