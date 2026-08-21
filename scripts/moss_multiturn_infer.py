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
import soundfile
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
    parser.add_argument("--loop-detect", choices=["none", "reset", "regen", "ratio"], default="none",
                        help="on audio-loop detection: clear the sliding window (reset) or also "
                             "regenerate the looping turn once at higher temperature (regen)")
    parser.add_argument("--sliding-window", type=int, default=0,
                        help="keep the last N-1 completed turns as context and rebuild the prompt "
                             "every turn (reset_cache each turn) instead of one growing session")
    parser.add_argument("--pin-first-turn", action="store_true",
                        help="sliding mode only: always keep the talk's first turn at the head of "
                             "the window (window = turn_0 + last N-2). Diagnostic for the silence-"
                             "anchor hypothesis (台账 4.-9 / 4.-10): every training row starts at "
                             "frame 0 of a synthesized passage, so its first context turn carries a "
                             "silence->speech onset; a sliding window never does. Turn 0 is the one "
                             "turn in a talk that does start from silence, so pinning it restores "
                             "the anchor at inference time without retraining.")
    # note (luojiaxuan): 2026-08-20 turn-0 启动修复（ChatGPT 审计 #2）：微型
    # 首轮（如"大家好，这是"）在无会话先验 + 15s 预算下会触发话语先验幻觉
    # （编播客开场白并复读）。首轮不足 min-tokens 且未到句末时并入后续增量
    # 再开口；首轮 runaway 下限单独收紧。置 0 关闭合并。
    parser.add_argument("--first-turn-min-tokens", type=int, default=12)
    parser.add_argument("--first-turn-floor-s", type=float, default=3.0)
    # note (luojiaxuan): 2026-08-21 句级缓冲合成——"断断续续"的治本方向：
    # 增量攒到句末标点（或 --sentence-merge-max-chars 上限）再合成，接缝
    # 从每 turn 一条降到每句一条且落在自然停顿处。改变延迟语义，属科学
    # 参数，显式 opt-in；canonical 评测维持 turn 级。
    parser.add_argument("--sentence-merge", action="store_true")
    parser.add_argument("--sentence-merge-max-chars", type=int, default=60)
    # note (luojiaxuan): 接缝交叉淡化实验后被用户否决（2026-08-21："治标
    # 不治本，听起来也很奇怪"）——默认关闭，仅留作诊断选项。治本方向是
    # 句级缓冲合成（--sentence-merge）。
    parser.add_argument("--seam-crossfade-ms", type=float, default=0.0)
    parser.add_argument("--soft-reset-keep", type=int, default=3,
                        help="sliding mode: when the window reaches its cap, shrink it to the most "
                             "recent N turns instead of sliding one-out-one-in. Context length then "
                             "cycles N..cap — periodically short like reset, but never empty, so no "
                             "hard boundary / timbre jump. THIS IS THE DEFAULT (N=3); pass 0 to opt "
                             "out and get a constant-length window that evicts one turn per step. "
                             "Why it is the default: the constant window changes its prompt prefix "
                             "every single turn, so the TTS KV cache is never reusable — measured "
                             "~478 prompt rows re-prefilled to decode ~24 new rows, a 20:1 blowup. "
                             "Soft reset only appends while growing (prompt_t is a strict prefix of "
                             "prompt_t+1, since a turn's leading break equals the previous header), "
                             "so the cache stays valid and is invalidated once per cycle instead of "
                             "once per turn — roughly an order of magnitude less prefill work. "
                             "Quality caveat: under the GPT-ASR op N=3 beat the constant window "
                             "(paired XCOMET t=+9.7); under the Qwen3-ASR canonical op it loses "
                             "(BLEU 26.5 vs 30.3). See 台账 4.-12 / 4.-16 before quoting either.")
    parser.add_argument("--min-frames-per-char", type=float, default=0.0,
                        help="sliding mode only: regenerate a turn once if it produced fewer than "
                             "this many codec frames per spoken character (0 = off). Diagnosed "
                             "failure mode: in a continuous session short turns get swallowed by "
                             "the previous utterance's prosody — sliding emits 11x more zero-frame "
                             "turns than session-reset. 1.5 sits below reset's 5th percentile (2.0)")
    parser.add_argument("--min-chars-for-short-check", type=int, default=3,
                        help="skip the short-turn check for texts below this many spoken chars")
    parser.add_argument("--reset-carry-seconds", type=float, default=0.0,
                        help="in reset mode, carry the previous session's last N seconds of codes "
                             "into the next session as a short prosodic anchor (0 = hard reset)")
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
        # note (luojiaxuan): torchaudio 2.11 的 load() 走 torchcodec，需要计算节点
        # 未必装的系统 FFmpeg 共享库（Tilde 上就没有）。参考音频只是普通 wav，
        # 用 soundfile 读即可，顺便让这个脚本能在 Tilde 上直接跑。
        data, sr = soundfile.read(args.fixed_ref, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)
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

    def history_turn_rows(text: str, codes: np.ndarray, prefill: str) -> np.ndarray:
        # note (luojiaxuan): same completed-turn layout as the finetuning packer
        # (text channel padded with <|text_pad|>, audio delayed by delay_tokens_len,
        # BOS before / EOS after the codes); zero-frame turns keep BOS+EOS adjacent.
        # 2026-08-20 修复（ChatGPT 审计 #2 发现）：重建窗口的最老一轮此前以
        # leading_break=False 直贴 ensemble，缺 "<|im_start|>assistant\n" 头，
        # 与训练排布不符。现在由调用方显式传 prefill：首个历史轮传
        # "<|im_start|>assistant\n"，其余传 "<|im_end|>\n<|im_start|>assistant\n"。
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

    def book1_ngrams(codes: np.ndarray, n: int = 8) -> set[tuple[int, ...]]:
        seq = codes[:, 0].tolist()
        return {tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)}

    def ratio_loop_detected(window_hist: list[tuple[str, np.ndarray]], codes: np.ndarray, text: str) -> bool:
        # note (luojiaxuan): loops emit audio without advancing text; detect via
        # rolling audio-seconds-per-spoken-char over the last few turns.
        frames = codes.shape[0] + sum(c.shape[0] for _, c in window_hist[-3:])
        chars = spoken_chars(text) + sum(spoken_chars(t) for t, _ in window_hist[-3:])
        if chars < 8:
            return False
        return (frames / 12.5) / chars > 0.42

    def loop_detected(codes: np.ndarray, prev: np.ndarray | None) -> bool:
        # note (luojiaxuan): phrase loops recur as long book-1 n-grams either
        # inside the turn or against the previous turn's codes.
        if codes.shape[0] >= 16:
            seq = codes[:, 0].tolist()
            grams = [tuple(seq[i : i + 8]) for i in range(len(seq) - 7)]
            if len(set(grams)) / max(1, len(grams)) < 0.6:
                return True
        if prev is not None and prev.shape[0] >= 8 and codes.shape[0] >= 8:
            cur = book1_ngrams(codes)
            if cur and len(cur & book1_ngrams(prev)) / len(cur) > 0.5:
                return True
        return False

    def window_prompt_ids(history: list[tuple[str, np.ndarray]]) -> np.ndarray:
        parts = [ensemble]
        for idx, (h_text, h_codes) in enumerate(history):
            parts.append(history_turn_rows(
                h_text, h_codes,
                prefill="<|im_end|>\n<|im_start|>assistant\n" if idx > 0
                else "<|im_start|>assistant\n",
            ))
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

    # note (luojiaxuan): 韵律锚点。硬 reset 每 11 轮把上下文清零，代价是边界
    # 处丢失音色/语调的衔接。这里只把上一段结尾的极短一截 codes 带过去当锚点
    # ——足以接住韵律，但太短、撑不起一个音频循环重新自我强化（保留整轮历史
    # 会重开污染通道，实测比硬切差 6-8 BLEU，见实验台账 4.1）。
    carry: tuple[str, np.ndarray] | None = None

    SENTENCE_FINAL = "。！？!?…"

    for processed, row in enumerate(rows, 1):
        row_id = str(row["row_id"])
        segments = row["segments"]
        if args.sentence_merge and segments:
            grouped, cur_text, cur_id, cur_n = [], "", None, 0
            for s in segments:
                if cur_id is None:
                    cur_id = s.get("id")
                cur_text += s["text"]
                cur_n += 1
                stripped = cur_text.rstrip()
                if (stripped and stripped[-1] in SENTENCE_FINAL) or \
                        len(cur_text) >= args.sentence_merge_max_chars:
                    grouped.append({"id": cur_id, "text": cur_text, "merged_turns": cur_n})
                    cur_text, cur_id, cur_n = "", None, 0
            if cur_text:
                grouped.append({"id": cur_id, "text": cur_text, "merged_turns": cur_n})
            segments = grouped
        if args.first_turn_min_tokens > 0 and segments:
            # note (luojiaxuan): 首轮缓冲——合并到 ≥N token 或句末为止
            merged_text, j = "", 0
            while j < len(segments):
                merged_text += segments[j]["text"]
                j += 1
                enough = len(tokenizer(merged_text)["input_ids"]) >= args.first_turn_min_tokens
                stripped = merged_text.rstrip()
                at_boundary = bool(stripped) and stripped[-1] in SENTENCE_FINAL
                if enough or at_boundary:
                    break
            if j > 1:
                segments = (
                    [{"id": segments[0].get("id"), "text": merged_text,
                      "merged_turns": j}] + segments[j:]
                )
        turn_results = []
        last_turn: tuple[str, np.ndarray] | None = None
        row_pcm: list[np.ndarray] = []
        failure = None
        window: list[tuple[str, np.ndarray]] = []
        # note (luojiaxuan): 2026-08-20 真 KV 复用（ChatGPT 审计 #2：此前
        # soft 模式每轮 reset_cache=True 全重建，"缩窗省 prefill"从未落地）。
        # kv_valid=True 表示 KV 里就是当前 window 的连续轨迹，可只追加
        # next_turn_ids 继续；缩窗/runaway/regen 任何一种破坏连续性都置 False
        # 并整窗重建。上游 prefill 原生支持 past_key_values 拼接。
        kv_valid = False
        # note (luojiaxuan): 第一个完成的 turn 会被记成锚点。它跨 loop-reset
        # 存活（reset 只清 window），因为它代表的是"从静音起音"这个形态，
        # 不是最近上下文。
        anchor_turn: tuple[str, np.ndarray] | None = None
        inferencer.reset_generation_state(keep_cache=False)
        with torch.inference_mode():
            for k, seg in enumerate(segments):
                text = seg["text"]
                floor_s = args.first_turn_floor_s if k == 0 else args.min_runaway_floor_s
                budget_frames = int(
                    max(floor_s, spoken_chars(text) * args.max_seconds_per_char) * 12.5
                )
                if args.sliding_window > 0:
                    if args.soft_reset_keep > 0 and kv_valid:
                        # 增长期纯追加：只送新轮 header，KV/mask 原地延续
                        session.reset_turn(
                            input_ids=next_turn_ids,
                            include_system_prompt=False,
                            reset_cache=False,
                        )
                    else:
                        session.reset_turn(
                            input_ids=window_prompt_ids(window),
                            include_system_prompt=False,
                            reset_cache=True,
                        )
                        kv_valid = args.soft_reset_keep > 0
                else:
                    if k == 0 and carry is not None:
                        anchor_ids = np.concatenate(
                            [
                                ensemble,
                                history_turn_rows(carry[0], carry[1],
                                                  prefill="<|im_start|>assistant\n"),
                                header_grid("<|im_end|>\n<|im_start|>assistant\n"),
                            ],
                            axis=0,
                        )
                    else:
                        anchor_ids = first_turn_ids
                    session.reset_turn(
                        input_ids=anchor_ids if k == 0 else next_turn_ids,
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
                runaway_skipped = False
                if not ok:
                    if args.sliding_window > 0:
                        # note (luojiaxuan): in sliding-window mode a runaway
                        # turn is truncated at its budget, the window is
                        # cleared (detox), and the row continues — a whole
                        # talk must not die on one bad turn.
                        runaway_skipped = True
                        window = []
                        kv_valid = False
                    else:
                        failure = f"turn{k} runaway: frames>{budget_frames}"
                        break
                turn_audio = (
                    np.concatenate(turn_pcm) if turn_pcm else np.zeros(0, dtype=np.float32)
                )
                if runaway_skipped:
                    turn_audio = turn_audio[: int(budget_frames / 12.5 * codec_sr)]
                # note (luojiaxuan): 短 turn 被吞并的护栏。滑窗模式下模型会把很短的
                # turn 并进上一句的韵律流里，产出畸短甚至零帧音频（实测滑窗零帧
                # turn 34 个 vs reset 3 个）。这里就地重生成一次；窗口重建、缓存
                # 重置，所以被吞的那次尝试不会留在上下文里。
                short_regen = False
                n_spoken = spoken_chars(text)
                if (args.sliding_window > 0 and args.min_frames_per_char > 0
                        and not runaway_skipped
                        and n_spoken >= args.min_chars_for_short_check
                        and turn_frames < args.min_frames_per_char * n_spoken):
                    short_regen = True
                    session.reset_turn(
                        input_ids=window_prompt_ids(window),
                        include_system_prompt=False,
                        reset_cache=True,
                    )
                    kv_valid = args.soft_reset_keep > 0
                    decoder = AudioStreamDecoder(
                        codec, chunk_frames=3, overlap_frames=0,
                        decode_kwargs={"chunk_duration": -1}, device=device,
                    )
                    turn_frames = 0
                    turn_pcm = []
                    turn_codes = []
                    with codec.streaming(batch_size=1):
                        ok_s = consume(session.push_text(text)) and consume(session.end_text())
                        while ok_s:
                            frames_s = session.drain(max_steps=1)
                            if not frames_s:
                                break
                            ok_s = consume(frames_s)
                            if session.inferencer.is_finished:
                                break
                        final_s = decoder.flush()
                        if final_s is not None and final_s.numel():
                            turn_pcm.append(final_s.detach().float().cpu().numpy().reshape(-1))
                    turn_audio = (
                        np.concatenate(turn_pcm) if turn_pcm else np.zeros(0, dtype=np.float32)
                    )

                row_pcm.append(turn_audio)
                if args.sliding_window == 0 and args.reset_carry_seconds > 0 and not runaway_skipped:
                    last_turn = (
                        text,
                        torch.cat(turn_codes, dim=0).numpy().astype(np.int64)
                        if turn_codes
                        else np.zeros((0, channels), dtype=np.int64),
                    )
                if args.sliding_window > 0 and runaway_skipped:
                    # polluted codes stay out of the (already cleared) window
                    turn_results.append(
                        {
                            "segment_id": seg.get("id"),
                            "text": text,
                            "duration_s": round(len(turn_audio) / codec_sr, 3),
                            "frames": turn_frames,
                            "runaway_skipped": True,
                        }
                    )
                    continue
                if args.sliding_window > 0:
                    codes_np = (
                        torch.cat(turn_codes, dim=0).numpy().astype(np.int64)
                        if turn_codes
                        else np.zeros((0, 16), dtype=np.int64)
                    )
                    prev_codes = window[-1][1] if window else None
                    trigger = False
                    if args.loop_detect == "ratio":
                        trigger = ratio_loop_detected(window, codes_np, text)
                    elif args.loop_detect != "none":
                        trigger = loop_detected(codes_np, prev_codes)
                    if trigger:
                        turn_results.append(
                            {"segment_id": seg.get("id"), "loop_detected": True}
                        ) if False else None
                        window = []
                        kv_valid = False
                        if args.loop_detect == "regen" and not getattr(seg, "_retried", False):
                            # regenerate this turn once with a clean window
                            session.temperature = min(1.0, args.temperature + 0.2)
                            session.reset_turn(
                                input_ids=window_prompt_ids(window),
                                include_system_prompt=False,
                                reset_cache=True,
                            )
                            kv_valid = args.soft_reset_keep > 0
                            decoder2 = AudioStreamDecoder(
                                codec, chunk_frames=3, overlap_frames=0,
                                decode_kwargs={"chunk_duration": -1}, device=device,
                            )
                            turn_frames = 0
                            turn_pcm = []
                            turn_codes = []
                            decoder = decoder2
                            with codec.streaming(batch_size=1):
                                ok2 = consume(session.push_text(text)) and consume(session.end_text())
                                while ok2:
                                    frames = session.drain(max_steps=1)
                                    if not frames:
                                        break
                                    ok2 = consume(frames)
                                    if session.inferencer.is_finished:
                                        break
                                final2 = decoder.flush()
                                if final2 is not None and final2.numel():
                                    turn_pcm.append(final2.detach().float().cpu().numpy().reshape(-1))
                            session.temperature = args.temperature
                            turn_audio = (
                                np.concatenate(turn_pcm) if turn_pcm else np.zeros(0, dtype=np.float32)
                            )
                            row_pcm[-1] = turn_audio
                            codes_np = (
                                torch.cat(turn_codes, dim=0).numpy().astype(np.int64)
                                if turn_codes
                                else np.zeros((0, 16), dtype=np.int64)
                            )
                            turn_results[-1]["duration_s"] = round(len(turn_audio) / codec_sr, 3)
                            turn_results[-1]["regenerated"] = True
                        turn_results[-1]["loop_reset"] = True
                    window.append((text, codes_np))
                    if args.pin_first_turn and anchor_turn is not None:
                        # note (luojiaxuan): 诊断模式，优先级最高——它与软 reset
                        # 互斥，而软 reset 现在是默认值，不显式排前面就永远进不来。
                        tail = max(0, args.sliding_window - 2)
                        recent = [t for t in window if t is not anchor_turn][-tail:]
                        window = [anchor_turn] + recent
                    elif args.soft_reset_keep > 0:
                        # note (luojiaxuan): 软 reset（默认）——长满就缩到最近 N 个，
                        # 未满就整窗保留（每轮只 append 一个，长度最多超上限 1）。
                        # 未满的那些轮是纯追加，prompt 严格扩展上一轮，KV cache 可复用；
                        # 只有缩窗那一刻前缀变化、需要重建。
                        if len(window) > args.sliding_window - 1:
                            window = window[-args.soft_reset_keep :]
                            kv_valid = False  # 缩窗即前缀变化，唯一重建点
                    else:
                        # note (luojiaxuan): --soft-reset-keep 0 的 opt-out：恒定长度
                        # 窗口，每轮挤掉最老的一个 turn。前缀每轮都变，KV cache 全废。
                        window = window[-(args.sliding_window - 1) :]
                    if anchor_turn is None and window:
                        anchor_turn = window[0]
                turn_results.append(
                    {
                        "segment_id": seg.get("id"),
                        "text": text,
                        "duration_s": round(len(turn_audio) / codec_sr, 3),
                        "frames": turn_frames,
                        **({"short_regen": True} if short_regen else {}),
                    }
                )

        if args.sliding_window == 0 and args.reset_carry_seconds > 0:
            # note (luojiaxuan): 只留尾部 N 秒的 codes，文本按帧数比例截同样
            # 长度的尾巴，避免 text/audio 长度严重错配落到训练分布之外。
            # runaway/失败的行不传递锚点，免得把坏音频带进下一段。
            if failure is not None or last_turn is None:
                carry = None
            else:
                keep = int(args.reset_carry_seconds * 12.5)
                c_text, c_codes = last_turn
                if c_codes.shape[0] <= keep:
                    carry = (c_text, c_codes)
                else:
                    frac = keep / c_codes.shape[0]
                    n_ch = max(1, int(round(len(c_text) * frac)))
                    carry = (c_text[-n_ch:], c_codes[-keep:])

        record = {
            "row_id": row_id,
            "split": row.get("split"),
            "num_segments": len(segments),
            "turns": turn_results,
            "failure": failure,
        }
        if failure is None:
            ov = int(args.seam_crossfade_ms / 1000.0 * codec_sr)
            if ov > 0 and len(row_pcm) > 1:
                merged = row_pcm[0].astype(np.float32, copy=True)
                for piece in row_pcm[1:]:
                    p = piece.astype(np.float32)
                    n = min(ov, len(merged), len(p))
                    if n > 0:
                        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
                        merged[-n:] = merged[-n:] * (1.0 - ramp) + p[:n] * ramp
                        merged = np.concatenate([merged, p[n:]])
                    else:
                        merged = np.concatenate([merged, p])
                audio = np.clip(merged, -1.0, 1.0)
            else:
                audio = np.clip(
                    np.concatenate(row_pcm) if row_pcm else np.zeros(1, dtype=np.float32),
                    -1.0, 1.0,
                )
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
