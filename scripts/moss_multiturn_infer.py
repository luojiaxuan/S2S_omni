#!/usr/bin/env python3
"""Multi-turn incremental MOSS-TTS-Realtime inference for InfiniSST rows.

Matches the v2 training format: one session per row, a fixed voice prompt in
the system ensemble, then assistant-only turns whose KV cache is retained, so
segment k is generated with segments 1..k-1 (text + generated audio) as
history. Turn prompts replicate finetuning/dataset.py exactly:
turn 0 = ensemble + "<|im_start|>assistant\\n", later turns =
"<|im_end|>\\n<|im_start|>assistant\\n".
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--codec-path", default="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    parser.add_argument("--moss-tts-root", default="/data/MOSS-TTS/moss_tts_realtime")
    parser.add_argument("--fixed-ref", required=True)
    parser.add_argument("--rows-jsonl", required=True, help="v2 rows format: {row_id, segments:[{id,text}...]}")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--summary-jsonl", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--repetition-window", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=8000)
    parser.add_argument("--max-seconds-per-char", type=float, default=0.6)
    parser.add_argument("--min-runaway-floor-s", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sliding-window", type=int, default=0,
                        help="keep the last N-1 completed turns as context and rebuild the prompt "
                             "every turn (reset_cache each turn) instead of one growing session")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def spoken_chars(text: str) -> int:
    return sum(1 for ch in text if unicodedata.category(ch)[0] in {"L", "N"})


def main() -> None:
    args = parse_args()
    sys.path.insert(0, args.moss_tts_root)
    from transformers import AutoModel, AutoTokenizer

    from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
    from mossttsrealtime.processing_mossttsrealtime import MossTTSRealtimeProcessor
    from mossttsrealtime.streaming_mossttsrealtime import (
        AudioStreamDecoder,
        MossTTSRealtimeInference,
        MossTTSRealtimeStreamingSession,
    )

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor = MossTTSRealtimeProcessor(tokenizer)
    model = (
        MossTTSRealtime.from_pretrained(
            args.model_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )
    codec = AutoModel.from_pretrained(args.codec_path, trust_remote_code=True).eval().to(device)
    codec_sr = int(getattr(codec.config, "sampling_rate", 24000))
    codebook_size = int(getattr(codec.config, "codebook_size", 1024))

    with torch.inference_mode():
        wav, sr = torchaudio.load(args.fixed_ref)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != codec_sr:
            wav = torchaudio.functional.resample(wav, sr, codec_sr)
        prompt_tokens = codec.encode(wav.unsqueeze(0).to(device))["audio_codes"].squeeze(1).cpu().numpy()

    def header_grid(text: str) -> np.ndarray:
        ids = tokenizer.encode(text)
        grid = np.full((len(ids), processor.channels + 1), processor.audio_channel_pad, dtype=np.int64)
        grid[:, 0] = ids
        return grid

    ensemble = processor.make_ensemble(prompt_tokens)
    first_turn_ids = np.concatenate([ensemble, header_grid("<|im_start|>assistant\n")], axis=0)
    next_turn_ids = header_grid("<|im_end|>\n<|im_start|>assistant\n")

    channels = processor.channels
    pad = processor.audio_channel_pad
    delay = processor.delay_tokens_len

    def history_turn_rows(text: str, codes: np.ndarray, leading_break: bool) -> np.ndarray:
        # note (luojiaxuan): same completed-turn layout as the finetuning packer
        # (text channel padded with <|text_pad|>, audio delayed by delay_tokens_len,
        # BOS before / EOS after the codes); zero-frame turns keep BOS+EOS adjacent.
        prefill = "<|im_end|>\n<|im_start|>assistant\n" if leading_break else ""
        text_tokens = tokenizer(text)["input_ids"]
        start = len(tokenizer(prefill)["input_ids"]) if prefill else 0
        audio_len = int(codes.shape[0])
        if len(text_tokens) >= delay:
            padded = audio_len + delay - len(text_tokens) + 1
            ch1 = tokenizer(prefill + text + "<|text_pad|>" * max(0, padded))["input_ids"]
            rows = np.full((len(ch1), channels + 1), pad, dtype=np.int64)
            rows[:, 0] = ch1
            a0 = start + delay
            rows[a0 : a0 + audio_len, 1:] = codes
            rows[a0 - 1, 1] = 1025
            rows[a0 + audio_len, 1] = 1026
        else:
            padded = audio_len + 1
            ch1 = tokenizer(prefill + text + "<|text_pad|>" * padded)["input_ids"]
            rows = np.full((len(ch1), channels + 1), pad, dtype=np.int64)
            rows[:, 0] = ch1
            if audio_len:
                rows[-(audio_len + 1) : -1, 1:] = codes
            rows[-(audio_len + 2), 1] = 1025
            rows[-1, 1] = 1026
        return rows

    def window_prompt_ids(history: list[tuple[str, np.ndarray]]) -> np.ndarray:
        parts = [ensemble]
        for idx, (h_text, h_codes) in enumerate(history):
            parts.append(history_turn_rows(h_text, h_codes, leading_break=idx > 0))
        parts.append(
            header_grid("<|im_end|>\n<|im_start|>assistant\n" if history else "<|im_start|>assistant\n")
        )
        return np.concatenate(parts, axis=0)

    inferencer = MossTTSRealtimeInference(model, tokenizer, max_length=args.max_length)
    audio_eos_token = int(getattr(inferencer, "audio_eos_token", 1026))
    session = MossTTSRealtimeStreamingSession(
        inferencer,
        processor,
        codec=codec,
        codec_sample_rate=codec_sr,
        codec_encode_kwargs={},
        prefill_text_len=processor.delay_tokens_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=True,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
    )
    session.set_voice_prompt_tokens(prompt_tokens)

    def sanitize(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() == 3:
            tokens = tokens[0]
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.numel() == 0:
            return tokens
        eos_rows = (tokens[:, 0] == audio_eos_token).nonzero(as_tuple=False)
        invalid = ((tokens < 0) | (tokens >= codebook_size)).any(dim=1)
        stop = None
        if eos_rows.numel() > 0:
            stop = int(eos_rows[0].item())
        if invalid.any():
            inv = int(invalid.nonzero(as_tuple=False)[0].item())
            stop = inv if stop is None else min(stop, inv)
        return tokens[:stop] if stop is not None else tokens

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_jsonl)
    done = set()
    if summary_path.exists():
        for line in summary_path.open(encoding="utf-8"):
            line = line.strip()
            if line:
                done.add(json.loads(line)["row_id"])

    rows = []
    with Path(args.rows_jsonl).open(encoding="utf-8") as handle:
        for idx, line in enumerate(l for l in handle if l.strip()):
            if idx % args.num_shards == args.shard_id:
                rows.append(json.loads(line))
    rows = [r for r in rows if str(r["row_id"]) not in done]
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    for processed, row in enumerate(rows, 1):
        row_id = str(row["row_id"])
        segments = row["segments"]
        turn_results = []
        row_pcm: list[np.ndarray] = []
        failure = None
        window: list[tuple[str, np.ndarray]] = []
        inferencer.reset_generation_state(keep_cache=False)
        with torch.inference_mode():
            for k, seg in enumerate(segments):
                text = seg["text"]
                budget_frames = int(
                    max(args.min_runaway_floor_s, spoken_chars(text) * args.max_seconds_per_char) * 12.5
                )
                if args.sliding_window > 0:
                    session.reset_turn(
                        input_ids=window_prompt_ids(window),
                        include_system_prompt=False,
                        reset_cache=True,
                    )
                else:
                    session.reset_turn(
                        input_ids=first_turn_ids if k == 0 else next_turn_ids,
                        include_system_prompt=False,
                        reset_cache=(k == 0),
                    )
                decoder = AudioStreamDecoder(
                    codec,
                    chunk_frames=3,
                    overlap_frames=0,
                    decode_kwargs={"chunk_duration": -1},
                    device=device,
                )
                turn_frames = 0
                turn_pcm: list[np.ndarray] = []
                turn_codes: list[torch.Tensor] = []

                def consume(frames_list) -> bool:
                    nonlocal turn_frames
                    for frame in frames_list:
                        tokens = sanitize(frame)
                        if tokens.numel() == 0:
                            continue
                        turn_frames += tokens.shape[0]
                        turn_codes.append(tokens.detach().cpu())
                        decoder.push_tokens(tokens.detach())
                        for chunk in decoder.audio_chunks():
                            if chunk.numel():
                                turn_pcm.append(chunk.detach().float().cpu().numpy().reshape(-1))
                    return turn_frames <= budget_frames

                ok = True
                with codec.streaming(batch_size=1):
                    ok = consume(session.push_text(text)) and consume(session.end_text())
                    while ok:
                        frames = session.drain(max_steps=1)
                        if not frames:
                            break
                        ok = consume(frames)
                        if session.inferencer.is_finished:
                            break
                    final = decoder.flush()
                    if final is not None and final.numel():
                        turn_pcm.append(final.detach().float().cpu().numpy().reshape(-1))
                if not ok:
                    failure = f"turn{k} runaway: frames>{budget_frames}"
                    break
                turn_audio = (
                    np.concatenate(turn_pcm) if turn_pcm else np.zeros(0, dtype=np.float32)
                )
                row_pcm.append(turn_audio)
                if args.sliding_window > 0:
                    codes_np = (
                        torch.cat(turn_codes, dim=0).numpy().astype(np.int64)
                        if turn_codes
                        else np.zeros((0, 16), dtype=np.int64)
                    )
                    window.append((text, codes_np))
                    window = window[-(args.sliding_window - 1) :]
                turn_results.append(
                    {
                        "segment_id": seg.get("id"),
                        "text": text,
                        "duration_s": round(len(turn_audio) / codec_sr, 3),
                        "frames": turn_frames,
                    }
                )

        record = {
            "row_id": row_id,
            "split": row.get("split"),
            "num_segments": len(segments),
            "turns": turn_results,
            "failure": failure,
        }
        if failure is None:
            audio = np.clip(np.concatenate(row_pcm) if row_pcm else np.zeros(1, dtype=np.float32), -1.0, 1.0)
            wav_path = out_dir / f"{row_id}.wav"
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(codec_sr)
                wf.writeframes((audio * 32767.0).astype(np.int16).tobytes())
            record["wav"] = str(wav_path)
            record["duration_s"] = round(len(audio) / codec_sr, 3)
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        if args.log_every > 0 and processed % args.log_every == 0:
            print(
                json.dumps({"processed": processed, "remaining": len(rows) - processed}),
                flush=True,
            )

    print(json.dumps({"shard": args.shard_id, "rows": len(rows), "done": True}), flush=True)


if __name__ == "__main__":
    main()
