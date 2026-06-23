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

from s2s_omni.omni_talker import Code2WavCapture, patch_qwen3_omni_no_split
from s2s_omni.rasst import DEFAULT_RASST_TRAIN, iter_rasst_rows, sanitize_id, write_mono_wav
from s2s_omni.textgrid import transcript_for_mfa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe native Qwen3-Omni text-to-audio path.")
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--input", default=DEFAULT_RASST_TRAIN)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--text")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=["repeat_prompt", "forced_assistant", "forced_decode"],
        default="forced_decode",
    )
    parser.add_argument("--speaker", default="Ethan")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--thinker-max-new-tokens", type=int, default=512)
    parser.add_argument("--talker-max-new-tokens", type=int, default=1024)
    parser.add_argument("--talker-min-new-tokens", type=int, default=0)
    parser.add_argument("--talker-do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--talker-repetition-penalty", type=float, default=1.05)
    parser.add_argument("--talker-temperature", type=float, default=0.9)
    parser.add_argument("--talker-top-k", type=int, default=50)
    parser.add_argument("--talker-top-p", type=float, default=1.0)
    return parser.parse_args()


def audio_to_numpy(wav: Any) -> np.ndarray:
    if hasattr(wav, "detach"):
        wav = wav.reshape(-1).detach().float().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    return np.clip(wav, -1.0, 1.0)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


def load_target_text(args: argparse.Namespace) -> tuple[str, str]:
    if args.text:
        return "manual", args.text
    for row in iter_rasst_rows(args.input):
        if row.index == args.row_index:
            return row.row_id, row.full_target_text
    raise SystemExit(f"row index not found: {args.row_index}")


def build_messages(target_text: str, mode: str) -> tuple[list[dict[str, str]], bool]:
    if mode == "forced_decode":
        return (
            [
                {
                    "role": "system",
                    "content": "You are a Mandarin text-to-speech engine. Speak the assistant text naturally.",
                },
            ],
            True,
        )
    if mode == "repeat_prompt":
        return (
            [
                {
                    "role": "system",
                    "content": "You are a Mandarin text-to-speech engine. Repeat exactly what the user asks you to speak.",
                },
                {
                    "role": "user",
                    "content": f"请逐字朗读下面的中文，不要改写，不要增删：\n{target_text}",
                },
            ],
            True,
        )
    return (
        [
            {
                "role": "system",
                "content": "You are a Mandarin text-to-speech engine. Speak the assistant text naturally.",
            },
            {"role": "user", "content": "Please speak the following Chinese text."},
            {"role": "assistant", "content": target_text},
        ],
        False,
    )


class ForcedTextLogitsProcessor:
    def __init__(self, *, prompt_len: int, forced_token_ids: list[int], eos_token_id: int) -> None:
        self.prompt_len = int(prompt_len)
        self.forced_token_ids = [int(token_id) for token_id in forced_token_ids]
        self.eos_token_id = int(eos_token_id)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        step = int(input_ids.shape[1]) - self.prompt_len
        token_id = self.forced_token_ids[step] if step < len(self.forced_token_ids) else self.eos_token_id
        scores[...] = -float("inf")
        scores[:, token_id] = 0.0
        return scores


class SuppressEosForInitialSteps:
    def __init__(self, *, eos_token_id: int, steps: int) -> None:
        self.eos_token_id = int(eos_token_id)
        self.steps = int(steps)
        self.start_len: int | None = None

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if self.start_len is None:
            self.start_len = int(input_ids.shape[1])
        generated = int(input_ids.shape[1]) - self.start_len
        if generated < self.steps:
            scores[:, self.eos_token_id] = -float("inf")
        return scores


def main() -> None:
    args = parse_args()
    import torch
    from transformers import LogitsProcessorList, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    row_id, target_text = load_target_text(args)
    safe_id = sanitize_id(f"{row_id}_{args.mode}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_qwen3_omni_no_split(Qwen3OmniMoeForConditionalGeneration)
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype="auto",
        device_map=args.device_map,
    ).eval()
    if not getattr(model, "has_talker", False):
        model.enable_talker()

    messages, add_generation_prompt = build_messages(target_text, args.mode)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    inputs = processor(text=text, return_tensors="pt", padding=True)
    device = getattr(model, "device", next(model.parameters()).device)
    inputs = inputs.to(device)
    prompt_len = int(inputs["input_ids"].shape[1])
    target_token_ids: list[int] = []
    thinker_extra_kwargs: dict[str, Any] = {}
    talker_extra_kwargs: dict[str, Any] = {}
    thinker_max_new_tokens = int(args.thinker_max_new_tokens)
    if args.mode == "forced_decode":
        target_token_ids = processor.tokenizer(target_text, add_special_tokens=False).input_ids
        eos_value = getattr(model.generation_config, "eos_token_id", None)
        if isinstance(eos_value, list):
            eos_value = eos_value[0] if eos_value else None
        eos_token_id = int(eos_value or getattr(processor.tokenizer, "eos_token_id", None) or 151645)
        thinker_extra_kwargs["thinker_logits_processor"] = LogitsProcessorList(
            [
                ForcedTextLogitsProcessor(
                    prompt_len=prompt_len,
                    forced_token_ids=target_token_ids,
                    eos_token_id=eos_token_id,
                )
            ]
        )
        thinker_max_new_tokens = max(thinker_max_new_tokens, len(target_token_ids) + 1)
    if args.talker_min_new_tokens > 0:
        talker_extra_kwargs["talker_logits_processor"] = LogitsProcessorList(
            [
                SuppressEosForInitialSteps(
                    eos_token_id=int(model.config.talker_config.codec_eos_token_id),
                    steps=int(args.talker_min_new_tokens),
                )
            ]
        )

    capture = Code2WavCapture(model.code2wav)
    try:
        with torch.inference_mode():
            generated, audio = model.generate(
                **inputs,
                thinker_return_dict_in_generate=True,
                thinker_max_new_tokens=thinker_max_new_tokens,
                thinker_do_sample=False,
                speaker=args.speaker,
                return_audio=True,
                talker_max_new_tokens=args.talker_max_new_tokens,
                talker_do_sample=args.talker_do_sample,
                talker_repetition_penalty=args.talker_repetition_penalty,
                talker_temperature=args.talker_temperature,
                talker_top_k=args.talker_top_k,
                talker_top_p=args.talker_top_p,
                **thinker_extra_kwargs,
                **talker_extra_kwargs,
            )
    finally:
        capture.restore()

    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    response_text = processor.batch_decode(
        sequences[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    wav = audio_to_numpy(audio)
    wav_path = out_dir / f"{safe_id}.wav"
    codes_path = out_dir / f"{safe_id}.npy"
    mfa_wav_path = out_dir / "mfa_corpus" / f"{safe_id}.wav"
    mfa_txt_path = out_dir / "mfa_corpus" / f"{safe_id}.txt"
    write_mono_wav(wav_path, wav, args.sample_rate)
    write_mono_wav(mfa_wav_path, wav, args.sample_rate)
    mfa_txt_path.parent.mkdir(parents=True, exist_ok=True)
    mfa_txt_path.write_text(transcript_for_mfa(target_text), encoding="utf-8")
    if capture.codes is not None:
        np.save(codes_path, capture.codes.squeeze(0).cpu().numpy().astype(np.int16, copy=False))

    duration_s = round(float(wav.size) / float(args.sample_rate), 6)
    record = {
        "row_id": row_id,
        "mode": args.mode,
        "accepted": bool(wav.size and capture.codes is not None),
        "target_text": target_text,
        "full_target_text": target_text,
        "response_text": response_text,
        "response_exact_match": response_text == target_text,
        "target_chars": len(target_text),
        "response_chars": len(response_text),
        "forced_text_tokens": len(target_token_ids),
        "thinker_max_new_tokens": thinker_max_new_tokens,
        "talker_min_new_tokens": int(args.talker_min_new_tokens),
        "wav_path": str(wav_path),
        "target_wav_path": str(wav_path),
        "duration_s": duration_s,
        "target_duration_s": duration_s,
        "codes_path": str(codes_path) if capture.codes is not None else None,
        "target_codes_path": str(codes_path) if capture.codes is not None else None,
        "codes_shape": list(capture.codes.squeeze(0).shape) if capture.codes is not None else None,
        "codec_frames": int(capture.codes.squeeze(0).shape[1]) if capture.codes is not None else 0,
        "codec_num_quantizers": int(capture.codes.squeeze(0).shape[0]) if capture.codes is not None else 0,
        "mfa_wav_path": str(mfa_wav_path),
        "mfa_txt_path": str(mfa_txt_path),
        "reject_reasons": [] if wav.size and capture.codes is not None else ["empty_wav_or_missing_codes"],
        "prompt_text": text,
    }
    (out_dir / f"{safe_id}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    append_jsonl(out_dir / "target_manifest.jsonl", record)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
