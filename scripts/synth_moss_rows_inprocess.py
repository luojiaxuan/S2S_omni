#!/usr/bin/env python3
"""In-process whole-passage synthesis of MOSS row requests (v7 data, 台账 4.-19).

与 generate_moss_realtime_long_targets.py 同一职责（干净训练目标合成），但
不走 serving——v2 时代的 sglang-omni MOSS 服务栈埋在 hyper00 旧容器的可写层
里，不可挂载复现。引擎混用的先例已经成立：v4 起训练上下文里的自历史音频
本来就是 in-process 生成的（Tilde），与 serving 合成的干净目标共存训出了
现役最优模型。采样用 MOSS 会话默认（temperature 0.8 / top_p 0.9 / top_k 50 /
repetition_penalty 1.1）；v2 的请求 payload 不带采样参数、即服务端默认，
两者同源于模型默认值。

行为对齐 generate 脚本：逐 group 独立一次性合成（fresh prompt），PCM 拼接
成整行 wav；runaway 预算 0.6 s/字（下限 15s），超限重试 --retries 次后整行
进 rejected。输出 row-raw jsonl 与 rejected jsonl 的 schema 与 generate 脚本
一致（下游 align_slice 只读 row_id/split/wav）。--num-shards/--shard-id 按行
索引切分，输出按 row_id 断点续跑。
"""
from __future__ import annotations

import argparse
import json
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np
import soundfile
import torch
import torchaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--rejected-jsonl", required=True, type=Path)
    parser.add_argument("--wav-dir", required=True, type=Path)
    parser.add_argument("--model-path", default="OpenMOSS-Team/MOSS-TTS-Realtime")
    parser.add_argument("--codec-path", default="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    parser.add_argument("--moss-tts-root", required=True)
    parser.add_argument("--fixed-ref", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=40960)
    parser.add_argument("--max-seconds-per-char", type=float, default=0.6)
    parser.add_argument("--min-runaway-floor-s", type=float, default=15.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--repetition-window", type=int, default=50)
    return parser.parse_args()


def spoken_chars(text: str) -> int:
    return sum(1 for ch in text if unicodedata.category(ch)[0] in ("L", "N"))


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> None:
    args = parse_args()
    import sys

    sys.path.insert(0, args.moss_tts_root)
    from transformers import AutoModel, AutoTokenizer
    from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
    from mossttsrealtime.processing_mossttsrealtime import MossTTSRealtimeProcessor
    from mossttsrealtime.streaming_mossttsrealtime import (
        AudioStreamDecoder,
        MossTTSRealtimeInference,
        MossTTSRealtimeStreamingSession,
    )

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = MossTTSRealtimeProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = (
        MossTTSRealtime.from_pretrained(args.model_path, trust_remote_code=True,
                                        torch_dtype=torch.bfloat16)
        .to(device).eval()
    )
    codec = AutoModel.from_pretrained(args.codec_path, trust_remote_code=True).eval().to(device)
    codec_sr = int(getattr(codec.config, "sampling_rate", 24000))

    data, sr = soundfile.read(args.fixed_ref, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != codec_sr:
        wav = torchaudio.functional.resample(wav, sr, codec_sr)
    with torch.inference_mode():
        prompt_tokens = codec.encode(wav.unsqueeze(0).to(device))["audio_codes"].squeeze(1).cpu().numpy()

    ensemble = processor.make_ensemble(prompt_tokens)

    def header_grid(text: str) -> np.ndarray:
        ids = tokenizer.encode(text)
        grid = np.full((len(ids), processor.channels + 1), processor.audio_channel_pad, dtype=np.int64)
        grid[:, 0] = ids
        return grid

    first_prompt = np.concatenate([ensemble, header_grid("<|im_start|>assistant\n")], axis=0)

    inferencer = MossTTSRealtimeInference(model, tokenizer, max_length=args.max_length)
    session = MossTTSRealtimeStreamingSession(
        inferencer, processor, codec=codec, codec_sample_rate=codec_sr,
        codec_encode_kwargs={}, prefill_text_len=processor.delay_tokens_len,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        do_sample=True, repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
    )
    session.set_voice_prompt_tokens(prompt_tokens)

    def synth_once(text: str, budget_s: float) -> tuple[np.ndarray | None, float]:
        """One-shot synthesis of one group; returns (pcm float32, duration_s)."""
        with torch.inference_mode():
            session.reset_turn(input_ids=first_prompt, include_system_prompt=False, reset_cache=True)
            decoder = AudioStreamDecoder(codec, chunk_frames=3, overlap_frames=0,
                                         decode_kwargs={"chunk_duration": -1}, device=device)
            frames, pcm_parts = 0, []
            budget_frames = int(budget_s * 12.5)

            def consume(frames_list) -> bool:
                nonlocal frames
                for frame in frames_list:
                    tok = frame if isinstance(frame, torch.Tensor) else torch.as_tensor(frame)
                    if tok.dim() == 3:
                        tok = tok[0]
                    if tok.dim() == 1:
                        tok = tok.unsqueeze(0)
                    if tok.numel() == 0:
                        continue
                    frames += int(tok.shape[0])
                    decoder.push_tokens(tok.detach())
                    for chunk in decoder.audio_chunks():
                        if chunk.numel():
                            pcm_parts.append(chunk.detach().float().cpu().numpy().reshape(-1))
                return frames <= budget_frames

            with codec.streaming(batch_size=1):
                ok = consume(session.push_text(text)) and consume(session.end_text())
                while ok and not session.inferencer.is_finished:
                    ok = consume(session.drain(max_steps=1))
                final = decoder.flush()
                if final is not None and final.numel():
                    pcm_parts.append(final.detach().float().cpu().numpy().reshape(-1))
            pcm = np.concatenate(pcm_parts) if pcm_parts else np.zeros(0, dtype=np.float32)
            return (pcm if ok else None), len(pcm) / codec_sr

    done = {str(r["row_id"]) for r in read_jsonl(args.output_jsonl)}
    done |= {str(r["row_id"]) for r in read_jsonl(args.rejected_jsonl)}
    rows = [
        row for idx, row in enumerate(read_jsonl(args.input_jsonl))
        if idx % args.num_shards == args.shard_id and str(row["row_id"]) not in done
    ]
    args.wav_dir.mkdir(parents=True, exist_ok=True)
    accepted = rejected = 0
    started_all = time.time()
    for processed, row in enumerate(rows, 1):
        row_id = str(row["row_id"])
        segments = row["segments"]
        pcm_all, group_results, failure = [], [], None
        for group_idx, group in enumerate(row["groups"]):
            text = "".join(segments[i]["text"] for i in group)
            budget_s = max(args.min_runaway_floor_s, spoken_chars(text) * args.max_seconds_per_char)
            attempt_durations = []
            group_ok = False
            for _attempt in range(args.retries + 1):
                t0 = time.perf_counter()
                pcm, duration = synth_once(text, budget_s)
                attempt_durations.append(round(duration, 3))
                if pcm is None:
                    failure = f"group{group_idx}: runaway duration {duration:.1f}s > budget {budget_s:.1f}s"
                    continue
                pcm_all.append(pcm)
                group_results.append({
                    "group": group_idx, "chars": len(text),
                    "duration_s": round(duration, 3),
                    "attempt_durations": attempt_durations,
                    "request_s": round(time.perf_counter() - t0, 3),
                })
                failure = None
                group_ok = True
                break
            if not group_ok:
                break
        if failure is not None:
            rejected += 1
            append_jsonl(args.rejected_jsonl,
                         {"row_id": row_id, "split": row.get("split"), "error": failure,
                          "groups": group_results})
        else:
            wav_path = args.wav_dir / f"{row_id}.wav"
            pcm16 = (np.clip(np.concatenate(pcm_all), -1.0, 1.0) * 32767.0).astype("<i2")
            with wave.open(str(wav_path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(codec_sr)
                out.writeframes(pcm16.tobytes())
            accepted += 1
            append_jsonl(args.output_jsonl, {
                "row_id": row_id, "split": row.get("split"), "wav": str(wav_path),
                "sample_rate": codec_sr,
                "duration_s": round(sum(g["duration_s"] for g in group_results), 3),
                "spoken_chars": row.get("spoken_chars"),
                "num_segments": len(segments), "groups": group_results,
                "fixed_ref": args.fixed_ref,
                "engine": "inprocess_transformers",
            })
        if args.log_every > 0 and processed % args.log_every == 0:
            elapsed = time.time() - started_all
            print(json.dumps({
                "shard": args.shard_id, "processed": processed,
                "accepted": accepted, "rejected": rejected,
                "remaining": len(rows) - processed,
                "rows_per_h": round(processed / max(elapsed, 1) * 3600, 1),
            }), flush=True)
    print(json.dumps({"shard": args.shard_id, "done": True,
                      "accepted": accepted, "rejected": rejected}))


if __name__ == "__main__":
    main()
