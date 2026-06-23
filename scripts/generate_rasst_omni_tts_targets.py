#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.omni_talker import (
    generate_talker_codes,
    patch_qwen3_omni_no_split,
    prepare_talker_condition,
)
from s2s_omni.rasst import (
    DEFAULT_RASST_DEV,
    DEFAULT_RASST_TRAIN,
    iter_rasst_rows,
    make_qwen_tts_messages,
    sanitize_id,
    write_mono_wav,
)
from s2s_omni.textgrid import transcript_for_mfa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full-sentence Qwen3-Omni target wav/codes for plain RASST baseline rows."
    )
    parser.add_argument("--input", default=DEFAULT_RASST_TRAIN)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--speaker", default="Ethan")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--codec-frame-rate", type=float, default=12.5)
    parser.add_argument("--hop-length", type=int, default=1920)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--talker-max-new-tokens", type=int, default=768)
    parser.add_argument("--talker-min-new-tokens", type=int, default=16)
    parser.add_argument("--talker-token-margin", type=int, default=8)
    parser.add_argument("--target-chars-per-second", type=float, default=4.0)
    parser.add_argument("--talker-do-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--talker-temperature", type=float, default=0.9)
    parser.add_argument("--talker-top-k", type=int, default=50)
    parser.add_argument("--talker-top-p", type=float, default=1.0)
    parser.add_argument("--talker-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--talker-suppress-primary-special", action="store_true")
    parser.add_argument("--talker-primary-from-sequences", action="store_true")
    parser.add_argument("--reject-primary-special", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-rejected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--write-mfa-corpus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write wav/txt files under output_dir/mfa_corpus for MFA alignment.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help=f"Shortcut for --input {DEFAULT_RASST_DEV}",
    )
    return parser.parse_args()


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


class OmniTtsTargetGenerator:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        self.torch = torch
        self.args = args
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

    def generate_one(self, target_text: str) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        max_new_tokens = self.effective_talker_max_new_tokens(target_text)
        messages = make_qwen_tts_messages(target_text)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        device = getattr(self.model, "device", next(self.model.parameters()).device)
        inputs = inputs.to(device)
        float_dtype = getattr(self.model, "dtype", None)
        if float_dtype is not None:
            for key in ["input_features", "pixel_values", "pixel_values_videos"]:
                if key in inputs:
                    inputs[key] = inputs[key].to(dtype=float_dtype)
        with torch.inference_mode():
            condition = prepare_talker_condition(self.model, inputs, speaker=self.args.speaker)
            codes = generate_talker_codes(
                self.model,
                condition,
                max_new_tokens=max_new_tokens,
                do_sample=self.args.talker_do_sample,
                temperature=self.args.talker_temperature,
                top_k=self.args.talker_top_k,
                top_p=self.args.talker_top_p,
                repetition_penalty=self.args.talker_repetition_penalty,
                suppress_primary_special=self.args.talker_suppress_primary_special,
                primary_from_sequences=self.args.talker_primary_from_sequences,
            )
            self.validate_codes(codes)
            wav = self.model.code2wav.chunked_decode(codes, chunk_size=300, left_context_size=25)
        if codes.ndim == 3 and codes.shape[0] == 1:
            codes = codes[0]
        return audio_to_numpy(wav), codes.detach().cpu().to(torch.int16).numpy()

    def validate_codes(self, codes: Any) -> None:
        torch = self.torch
        if codes.ndim == 3 and codes.shape[0] == 1:
            codes_2d = codes[0].detach().cpu().long()
        else:
            codes_2d = codes.detach().cpu().long()
        if codes_2d.ndim != 2:
            raise RuntimeError(f"expected generated codes [Q,T], got {tuple(codes_2d.shape)}")
        codebook = int(self.model.config.code2wav_config.codebook_size)
        primary_vocab = int(self.model.config.talker_config.text_config.vocab_size)
        residual_vocab = int(self.model.config.talker_config.code_predictor_config.vocab_size)
        eos = int(self.model.config.talker_config.codec_eos_token_id)
        q0 = codes_2d[0]
        forbidden_q0 = (q0 >= codebook) & (q0 < primary_vocab) & (q0 != eos)
        if self.args.reject_primary_special and bool(forbidden_q0.any()):
            bad = torch.unique(q0[forbidden_q0])[:16].tolist()
            raise RuntimeError(f"forbidden primary codec tokens from talker: {bad}")
        if codes_2d.shape[0] > 1:
            residual = codes_2d[1:]
            bad_residual = (residual < 0) | (residual >= residual_vocab)
            if bool(bad_residual.any()):
                bad = torch.unique(residual[bad_residual])[:16].tolist()
                raise RuntimeError(f"invalid residual codec tokens: {bad}")

    def effective_talker_max_new_tokens(self, target_text: str) -> int:
        speakable_chars = max(1, len(transcript_for_mfa(target_text).replace(" ", "")))
        estimated_frames = int(np.ceil(speakable_chars / self.args.target_chars_per_second * self.args.codec_frame_rate))
        capped = estimated_frames + int(self.args.talker_token_margin)
        return max(
            int(self.args.talker_min_new_tokens),
            min(int(self.args.talker_max_new_tokens), capped),
        )


def main() -> None:
    args = parse_args()
    if args.dev:
        args.input = DEFAULT_RASST_DEV
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")

    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "target_wav"
    codes_dir = output_dir / "target_codes"
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
        print(json.dumps({"selected": 0, "done": len(done)}, indent=2, ensure_ascii=False))
        return

    generator = OmniTtsTargetGenerator(args)
    accepted = 0
    rejected = 0
    for index, row in enumerate(selected, start=1):
        safe_id = sanitize_id(row.row_id)
        out_record: dict[str, Any] = {
            "row_id": row.row_id,
            "source_path": row.source_path,
            "row_index": row.index,
            "full_target_text": row.full_target_text,
            "effective_talker_max_new_tokens": generator.effective_talker_max_new_tokens(row.full_target_text),
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
            "speaker": args.speaker,
            "sample_rate": args.sample_rate,
            "codec_frame_rate": args.codec_frame_rate,
            "hop_length": args.hop_length,
        }
        try:
            wav, codes = generator.generate_one(row.full_target_text)
            if wav.size == 0:
                raise RuntimeError("empty generated wav")
            if codes.ndim != 2:
                raise RuntimeError(f"expected 2-D codes, got {codes.shape}")
            wav_path = wav_dir / f"{safe_id}.wav"
            codes_path = codes_dir / f"{safe_id}.npy"
            write_mono_wav(wav_path, wav, args.sample_rate)
            codes_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(codes_path, codes)
            out_record.update(
                {
                    "accepted": True,
                    "target_wav_path": str(wav_path),
                    "target_codes_path": str(codes_path),
                    "target_duration_s": round(float(wav.size) / float(args.sample_rate), 6),
                    "codec_frames": int(codes.shape[1]),
                    "codec_num_quantizers": int(codes.shape[0]),
                    "reject_reasons": [],
                }
            )
            if args.write_mfa_corpus:
                mfa_wav = mfa_dir / f"{safe_id}.wav"
                mfa_txt = mfa_dir / f"{safe_id}.txt"
                write_mono_wav(mfa_wav, wav, args.sample_rate)
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
        "speaker": args.speaker,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    (output_dir / f"summary_shard{args.shard_index}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
