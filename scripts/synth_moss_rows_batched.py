#!/usr/bin/env python3
"""Batched in-process whole-passage synthesis (v7 data, 台账 4.-19).

synth_moss_rows_inprocess.py 的 batch=1 会话式实现实测仅 ~104 行/时/卡
（38s 中位行 = ~480 步串行，H200 算力闲置）——与 gen_moss_self_history
注释里记录的同一个坑（"单流封装实测 ~38s/turn，全量跑不完"）。本脚本
移植它的批式引擎：batch 内 lockstep prefill/step，逐条按各自 EOS/预算
截断；工作单元是 (row, group)，池子按文本长度排序保证 batch 同质。

其余行为与 batch=1 版一致：整段 fresh-prompt 合成、runaway 预算 0.6s/字、
超限重试一次、行内任一 group 失败整行进 rejected；row-raw/rejected schema
不变。--done-glob 把 batch=1 版已完成的行并入断点集，两版产物可混用。
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import time
import unicodedata
import wave
from collections import deque
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
    parser.add_argument("--done-glob", nargs="*", default=[],
                        help="additional jsonl globs whose row_ids count as done")
    parser.add_argument("--model-path", default="OpenMOSS-Team/MOSS-TTS-Realtime")
    parser.add_argument("--codec-path", default="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    parser.add_argument("--moss-tts-root", required=True)
    parser.add_argument("--fixed-ref", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seconds-per-char", type=float, default=0.6)
    parser.add_argument("--min-runaway-floor-s", type=float, default=15.0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--repetition-window", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def spoken_chars(text: str) -> int:
    return sum(1 for ch in text if unicodedata.category(ch)[0] in ("L", "N"))


def read_jsonl(path: Path):
    if not Path(path).exists():
        return
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


class GroupItem:
    __slots__ = ("row_id", "group_idx", "text", "budget", "attempts")

    def __init__(self, row_id: str, group_idx: int, text: str, budget: int) -> None:
        self.row_id, self.group_idx, self.text = row_id, group_idx, text
        self.budget, self.attempts = budget, 0


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
    )

    device = torch.device(args.device)
    torch.manual_seed(args.seed + args.shard_id)
    torch.cuda.manual_seed_all(args.seed + args.shard_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor = MossTTSRealtimeProcessor(tokenizer)
    model = (
        MossTTSRealtime.from_pretrained(
            args.model_path, attn_implementation="sdpa",
            torch_dtype=torch.bfloat16, trust_remote_code=True)
        .to(device).eval()
    )
    codec = AutoModel.from_pretrained(args.codec_path, trust_remote_code=True).eval().to(device)
    codec_sr = int(getattr(codec.config, "sampling_rate", 24000))

    with torch.inference_mode():
        data, sr = soundfile.read(args.fixed_ref, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != codec_sr:
            wav = torchaudio.functional.resample(wav, sr, codec_sr)
        prompt_tokens = codec.encode(wav.unsqueeze(0).to(device))["audio_codes"].squeeze(1).cpu().numpy()

    channels = processor.channels
    pad = processor.audio_channel_pad
    delay = processor.delay_tokens_len
    ensemble = processor.make_ensemble(prompt_tokens)
    text_pad_id = tokenizer.convert_tokens_to_ids("<|text_pad|>")

    def header_grid(text: str) -> np.ndarray:
        ids = tokenizer.encode(text)
        grid = np.full((len(ids), channels + 1), pad, dtype=np.int64)
        grid[:, 0] = ids
        return grid

    first_prompt = np.concatenate([ensemble, header_grid("<|im_start|>assistant\n")], axis=0)

    inferencer = MossTTSRealtimeInference(model, tokenizer, max_length=40960)
    audio_eos = int(getattr(inferencer, "audio_eos_token", 1026))
    codebook_size = int(getattr(codec.config, "codebook_size", 1024))

    # ---- 断点集：本分片输出 + batch=1 版产物 + 其他 done-glob ----
    done: set[str] = set()
    for r in read_jsonl(args.output_jsonl):
        done.add(str(r["row_id"]))
    for r in read_jsonl(args.rejected_jsonl):
        done.add(str(r["row_id"]))
    for pattern in args.done_glob:
        for path in globmod.glob(pattern):
            for r in read_jsonl(Path(path)):
                done.add(str(r["row_id"]))

    rows = [
        row for idx, row in enumerate(read_jsonl(args.input_jsonl))
        if idx % args.num_shards == args.shard_id and str(row["row_id"]) not in done
    ]
    row_meta = {str(r["row_id"]): r for r in rows}

    items: list[GroupItem] = []
    for row in rows:
        segs = row["segments"]
        for gi, group in enumerate(row["groups"]):
            text = "".join(segs[i]["text"] for i in group)
            budget = int(max(args.min_runaway_floor_s,
                             spoken_chars(text) * args.max_seconds_per_char) * 12.5)
            items.append(GroupItem(str(row["row_id"]), gi, text, budget))
    # note (luojiaxuan): batch 内 lockstep 跑到最长预算，长度混杂会让短行陪跑；
    # 按文本长度排序，同 batch 预算相近，浪费最小。
    items.sort(key=lambda it: len(it.text))
    pool = deque(items)

    pending: dict[str, dict[int, dict]] = {}
    need: dict[str, int] = {str(r["row_id"]): len(r["groups"]) for r in rows}
    rejected_rows: set[str] = set()

    args.wav_dir.mkdir(parents=True, exist_ok=True)
    accepted = rejected = rounds = 0
    started = time.time()

    def finish_row(row_id: str) -> None:
        nonlocal accepted
        meta = row_meta[row_id]
        groups = [pending[row_id][gi] for gi in sorted(pending[row_id])]
        pcm16 = (np.clip(np.concatenate([g.pop("pcm") for g in groups]), -1.0, 1.0)
                 * 32767.0).astype("<i2")
        wav_path = args.wav_dir / f"{row_id}.wav"
        with wave.open(str(wav_path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(codec_sr)
            out.writeframes(pcm16.tobytes())
        append_jsonl(args.output_jsonl, {
            "row_id": row_id, "split": meta.get("split"), "wav": str(wav_path),
            "sample_rate": codec_sr,
            "duration_s": round(sum(g["duration_s"] for g in groups), 3),
            "spoken_chars": meta.get("spoken_chars"),
            "num_segments": len(meta["segments"]), "groups": groups,
            "fixed_ref": args.fixed_ref, "engine": "inprocess_batched",
        })
        accepted += 1
        del pending[row_id]

    def reject_row(row_id: str, error: str) -> None:
        nonlocal rejected
        append_jsonl(args.rejected_jsonl, {
            "row_id": row_id, "split": row_meta[row_id].get("split"), "error": error,
            "groups": [pending.get(row_id, {}).get(gi) for gi in
                       sorted(pending.get(row_id, {}))],
        })
        rejected += 1
        rejected_rows.add(row_id)
        pending.pop(row_id, None)

    inflight: list[GroupItem] = []
    while pool or inflight:
        while len(inflight) < args.batch_size and pool:
            item = pool.popleft()
            if item.row_id in rejected_rows:
                continue
            inflight.append(item)
        if not inflight:
            break

        text_ids = [tokenizer.encode(it.text, add_special_tokens=False) for it in inflight]
        budgets = [it.budget for it in inflight]
        inferencer.reset_generation_state(keep_cache=False)
        with torch.inference_mode():
            first = inferencer.prefill(
                input_ids=[first_prompt for _ in inflight],
                text_prefix_ids=[ids[:delay] if len(ids) >= delay else (ids or [text_pad_id])
                                 for ids in text_ids],
                device=device,
                temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                do_sample=True, repetition_penalty=args.repetition_penalty,
                repetition_window=args.repetition_window,
            )
            frames = [first]
            max_budget = max(budgets)
            step_i = 0
            while step_i < max_budget and not inferencer.is_finished:
                toks = []
                for b, ids in enumerate(text_ids):
                    pos = delay + step_i
                    toks.append(int(ids[pos]) if pos < len(ids) else text_pad_id)
                frames.append(inferencer.step(
                    toks,
                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    do_sample=True, repetition_penalty=args.repetition_penalty,
                    repetition_window=args.repetition_window,
                ))
                step_i += 1

        rounds += 1
        stacked = torch.stack(frames, dim=0).cpu().numpy()  # [T, B, channels]
        retired: list[GroupItem] = []
        for b, it in enumerate(inflight):
            seq = stacked[:, b, :]
            eos_pos = np.where(seq[:, 0] == audio_eos)[0]
            end = int(eos_pos[0]) if eos_pos.size else seq.shape[0]
            # sanitize：越界值（BOS/EOS 混入音频通道等）处一并截断
            invalid = np.where(((seq < 0) | (seq >= codebook_size)).any(axis=1))[0]
            if invalid.size:
                end = min(end, int(invalid[0]))
            runaway = (not eos_pos.size) or end > it.budget
            end = min(end, it.budget)
            codes = seq[:end].astype(np.int64)
            if runaway:
                it.attempts += 1
                if it.attempts <= 1:
                    pool.append(it)  # 重试一次
                else:
                    reject_row(it.row_id,
                               f"group{it.group_idx}: runaway > {it.budget} frames")
                retired.append(it)
                continue
            with torch.inference_mode(), codec.streaming(batch_size=1):
                dec = AudioStreamDecoder(codec, chunk_frames=3, overlap_frames=0,
                                         decode_kwargs={"chunk_duration": -1}, device=device)
                pcm_parts = []
                if codes.shape[0]:
                    dec.push_tokens(torch.from_numpy(codes).to(device))
                    for chunk in dec.audio_chunks():
                        if chunk.numel():
                            pcm_parts.append(chunk.detach().float().cpu().numpy().reshape(-1))
                tail = dec.flush()
                if tail is not None and tail.numel():
                    pcm_parts.append(tail.detach().float().cpu().numpy().reshape(-1))
            pcm = np.concatenate(pcm_parts) if pcm_parts else np.zeros(0, dtype=np.float32)
            pending.setdefault(it.row_id, {})[it.group_idx] = {
                "group": it.group_idx, "chars": len(it.text),
                "duration_s": round(len(pcm) / codec_sr, 3),
                "attempt_durations": [round(len(pcm) / codec_sr, 3)],
                "pcm": pcm,
            }
            if (it.row_id not in rejected_rows
                    and len(pending.get(it.row_id, {})) == need[it.row_id]):
                finish_row(it.row_id)
            retired.append(it)
        for it in retired:
            inflight.remove(it)

        if args.log_every and rounds % args.log_every == 0:
            elapsed = time.time() - started
            print(json.dumps({
                "shard": args.shard_id, "rounds": rounds,
                "accepted": accepted, "rejected": rejected,
                "pool_left": len(pool), "inflight": len(inflight),
                "rows_per_h": round(accepted / max(elapsed, 1) * 3600, 1),
            }), flush=True)

    # 尾账：有 group 永远没等齐的行（不应发生，防御性记录）
    for row_id in list(pending):
        reject_row(row_id, "incomplete groups at shutdown")
    print(json.dumps({"shard": args.shard_id, "done": True,
                      "accepted": accepted, "rejected": rejected}))


if __name__ == "__main__":
    main()
