#!/usr/bin/env python3
"""Generate closed-loop self-history audio codes for v4 scheduled sampling.

v3 的重复污染增强用的是手写的三种坏历史模式；v4 换成模型自己的真实错误
分布：对每一行多 turn 记录，用当前 checkpoint 以**滑动窗口闭环**方式逐 turn
生成——turn k 的 prompt 里放的是模型自己生成的 turn 0..k-1，而不是训练集里
base 模型一次合成再切片的干净历史。产出的 self codes 由
``build_moss_v4_dataset.py`` 按概率替换进上下文（标 ``context_only``，不进
loss），从而让训练分布对齐推理分布。

Prompt 布局与 ``scripts/moss_multiturn_infer.py`` 逐字节一致（也就是与
finetuning packer 一致）：ensemble(ref codes) + 最近 W-1 个 turn 的
``<|im_start|>assistant\\n text⊕codes <|im_end|>\\n`` + 新 turn header。

吞吐靠底层 ``MossTTSRealtimeInference`` 的 batch 能力：上层 streaming
session 是单流封装（实测 ~38s/turn，全量跑不完），这里改成一次 prefill 一
整批 turn，左 padding 对齐，逐帧 step 到各自 EOS。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unicodedata
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torchaudio

FRAME_RATE = 12.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--codec-path", default="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    parser.add_argument("--moss-tts-root", required=True)
    parser.add_argument("--fixed-ref", required=True)
    parser.add_argument("--records-jsonl", required=True,
                        help="multi-turn records: {id, conversations:[{role,text,audio_codes}], ...}")
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--heartbeat", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sliding-window", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-turns-per-row", type=int, default=0,
                        help="cap turns generated per row (0 = all); long sessions are expensive")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--repetition-window", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=8000)
    parser.add_argument("--max-seconds-per-char", type=float, default=0.6)
    parser.add_argument("--min-runaway-floor-s", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


def spoken_chars(text: str) -> int:
    return sum(1 for ch in text if unicodedata.category(ch)[0] in {"L", "N"})


class RowState:
    """One row in flight: which turn it is on and what it has generated so far."""

    def __init__(self, record: dict, max_turns: int):
        self.id = record["id"]
        turns = [t for t in record["conversations"] if t.get("role") == "assistant"]
        if max_turns > 0:
            turns = turns[:max_turns]
        self.texts = [t.get("text", "") for t in turns]
        self.gt_frames = [len(t.get("audio_codes") or []) for t in turns]
        self.turn_idx = 0
        self.history: list[tuple[str, np.ndarray]] = []  # self-generated
        self.self_codes: list[np.ndarray] = []
        self.runaway: list[bool] = []

    @property
    def done(self) -> bool:
        return self.turn_idx >= len(self.texts)

    def budget_frames(self, floor_s: float, sec_per_char: float) -> int:
        text = self.texts[self.turn_idx]
        return int(max(floor_s, spoken_chars(text) * sec_per_char) * FRAME_RATE)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, args.moss_tts_root)
    from transformers import AutoModel, AutoTokenizer

    from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
    from mossttsrealtime.processing_mossttsrealtime import MossTTSRealtimeProcessor
    from mossttsrealtime.streaming_mossttsrealtime import MossTTSRealtimeInference

    device = torch.device(args.device)
    torch.manual_seed(args.seed + args.shard_id)
    torch.cuda.manual_seed_all(args.seed + args.shard_id)

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

    with torch.inference_mode():
        wav, sr = torchaudio.load(args.fixed_ref)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != codec_sr:
            wav = torchaudio.functional.resample(wav, sr, codec_sr)
        prompt_tokens = codec.encode(wav.unsqueeze(0).to(device))["audio_codes"].squeeze(1).cpu().numpy()

    channels = processor.channels
    pad = processor.audio_channel_pad
    delay = processor.delay_tokens_len
    ensemble = processor.make_ensemble(prompt_tokens)

    def header_grid(text: str) -> np.ndarray:
        ids = tokenizer.encode(text)
        grid = np.full((len(ids), channels + 1), pad, dtype=np.int64)
        grid[:, 0] = ids
        return grid

    def history_turn_rows(text: str, codes: np.ndarray, leading_break: bool) -> np.ndarray:
        # note (luojiaxuan): identical completed-turn layout to the finetuning
        # packer and to moss_multiturn_infer.py; keep the three in lockstep.
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
    audio_eos = int(getattr(inferencer, "audio_eos_token", 1026))
    text_pad_id = int(inferencer.text_pad_id)

    # ---------------- load + shard + resume ----------------
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if out_path.exists():
        with out_path.open() as handle:
            for line in handle:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    continue

    records = []
    with open(args.records_jsonl) as handle:
        for i, line in enumerate(handle):
            if i % args.num_shards != args.shard_id:
                continue
            row = json.loads(line)
            if row["id"] in done_ids:
                continue
            records.append(row)
            if args.max_rows and len(records) >= args.max_rows:
                break
    print(json.dumps({"shard": args.shard_id, "pending_rows": len(records),
                      "already_done": len(done_ids)}), flush=True)

    pool = deque(RowState(r, args.max_turns_per_row) for r in records)
    pool = deque(st for st in pool if st.texts)
    inflight: list[RowState] = []
    sink = out_path.open("a")
    started = time.time()
    rows_done = 0
    turns_done = 0

    def emit(state: RowState) -> None:
        sink.write(json.dumps({
            "id": state.id,
            "turns": [
                {
                    "text": state.texts[i],
                    "self_audio_codes": state.self_codes[i].tolist(),
                    "self_frames": int(state.self_codes[i].shape[0]),
                    "gt_frames": state.gt_frames[i],
                    "runaway": bool(state.runaway[i]),
                }
                for i in range(len(state.self_codes))
            ],
        }, ensure_ascii=False) + "\n")
        sink.flush()

    while pool or inflight:
        while len(inflight) < args.batch_size and pool:
            inflight.append(pool.popleft())
        if not inflight:
            break

        prompts = [window_prompt_ids(st.history[-(args.sliding_window - 1):]
                                     if args.sliding_window > 1 else []) for st in inflight]
        text_ids = [tokenizer.encode(st.texts[st.turn_idx], add_special_tokens=False) for st in inflight]
        budgets = [st.budget_frames(args.min_runaway_floor_s, args.max_seconds_per_char) for st in inflight]

        inferencer.reset_generation_state(keep_cache=False)
        with torch.inference_mode():
            first = inferencer.prefill(
                input_ids=prompts,
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

        # [T, B, channels] -> per-row codes, cut at that row's EOS or budget
        stacked = torch.stack(frames, dim=0).cpu().numpy()
        finished: list[RowState] = []
        for b, st in enumerate(inflight):
            seq = stacked[:, b, :]
            eos_pos = np.where(seq[:, 0] == audio_eos)[0]
            end = int(eos_pos[0]) if eos_pos.size else seq.shape[0]
            runaway = not eos_pos.size or end > budgets[b]
            end = min(end, budgets[b])
            codes = seq[:end].astype(np.int64)
            st.self_codes.append(codes)
            st.runaway.append(runaway)
            st.history.append((st.texts[st.turn_idx], codes))
            st.turn_idx += 1
            turns_done += 1
            if st.done:
                emit(st)
                rows_done += 1
                finished.append(st)
        for st in finished:
            inflight.remove(st)

        if args.log_every and rows_done and rows_done % args.log_every == 0:
            elapsed = time.time() - started
            print(json.dumps({
                "shard": args.shard_id, "rows_done": rows_done, "turns_done": turns_done,
                "rows_left": len(pool) + len(inflight),
                "turns_per_s": round(turns_done / max(elapsed, 1e-6), 3),
            }), flush=True)
            if args.heartbeat:
                Path(args.heartbeat).write_text(json.dumps({
                    "ts": time.time(), "shard": args.shard_id,
                    "rows_done": rows_done, "turns_done": turns_done,
                    "rows_left": len(pool) + len(inflight),
                }))

    sink.close()
    print(json.dumps({"shard": args.shard_id, "status": "SHARD_DONE",
                      "rows_done": rows_done, "turns_done": turns_done,
                      "elapsed_s": round(time.time() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
