#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Qwen3-TTS voice-clone CLI.")
    parser.add_argument("--text", default=None, help="Text to synthesize. Defaults to S2S_TEXT.")
    parser.add_argument("--output", default=None, help="Output wav path. Defaults to S2S_OUTPUT_WAV.")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--ref-text", default=None)
    parser.add_argument(
        "--x-vector-only-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use only speaker embedding from ref audio; usually lower quality but no ref transcript needed.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    return parser.parse_args()


def torch_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    args = parse_args()
    text = args.text if args.text is not None else os.environ.get("S2S_TEXT", "")
    output = Path(args.output or os.environ.get("S2S_OUTPUT_WAV", "output.wav"))
    ref_audio = args.ref_audio or os.environ.get("S2S_REF_AUDIO") or os.environ.get("S2S_SOURCE_AUDIO")
    ref_text = args.ref_text or os.environ.get("S2S_REF_TEXT") or os.environ.get("S2S_SOURCE_TEXT")
    if not text.strip():
        raise ValueError("empty --text/S2S_TEXT")
    if not ref_audio:
        raise ValueError("Qwen3-TTS Base voice clone needs --ref-audio or S2S_SOURCE_AUDIO")
    if not ref_text and not args.x_vector_only_mode:
        raise ValueError("Qwen3-TTS Base voice clone needs --ref-text/S2S_SOURCE_TEXT unless x-vector-only")

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    started = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device_map,
        dtype=torch_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
    )
    load_s = time.perf_counter() - started
    synth_started = time.perf_counter()
    with torch.inference_mode():
        if args.x_vector_only_mode:
            prompt = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text or "",
                x_vector_only_mode=True,
            )
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=args.language,
                voice_clone_prompt=prompt,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=args.language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                max_new_tokens=args.max_new_tokens,
            )
    synth_s = time.perf_counter() - synth_started
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), wavs[0], sr)
    duration_s = len(wavs[0]) / float(sr)
    print(
        {
            "output": str(output),
            "sample_rate": sr,
            "duration_s": round(duration_s, 6),
            "model_load_s": round(load_s, 6),
            "synth_s": round(synth_s, 6),
        },
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
