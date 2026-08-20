# Audit bundle: why did v7's null-alignment rate double at 1.5× source speed?

You are auditing a speech-to-speech translation training/eval pipeline for bugs.
Everything needed is inlined below. Please look for: training-data construction
bugs, training launch bugs, inference/serving bugs, and evaluation bugs that
could produce the observed asymmetry. Also judge whether the observation is
explainable by measurement noise given the design.

## The observation

Cascade: InfiniSST (en→zh streaming S2T, fixed per-speed turn streams — the
SAME input turn files are fed to every TTS model, so S2T is controlled) →
MOSS-TTS-Realtime (finetuned, turn-by-turn TTS with a constant 10-turn sliding
context window). Scoring: self-hosted Qwen3-ASR-1.7B transcribes the target
speech → SEGALE sentence alignment (LaBSE+vecalign) → BLEU (null alignments
kept as empty hypotheses) + XCOMET-XL reference-based (nulls fixed to 0.0).
"null" = a reference sentence with no aligned hypothesis span (under-translation)
or vice versa.

Numbers (BLEU / XCOMET-ref / null rate, each cell = ONE generation + ONE scoring):

| model | train set | 1× | 1.25× | 1.5× |
|---|---|---|---|---|
| v6 | 36,529 rows | 30.32 / 0.586 / 8.1% | 30.28 / 0.619 / 6.7% | 31.21 / 0.609 / 5.2% (15/291) |
| v7 | v6 + 12,518 new rows | **31.03 / 0.652 / 2.0%** | 30.12 / 0.604 / 9.3% | 30.73 / 0.589 / **11.0% (35/317)** |

v7 = v6's exact training set + rows built from a new trajectory-aligned corpus
(6,385 passages, 157,887 turns, median turn 4–5 zh chars — much shorter turns
than the v6 data) + mid-passage-start copies of those rows. Same base model,
same hyperparameters, same trainer, global batch 15, 1 epoch, lr 1e-5.

The puzzle: v7 massively improves nulls at 1× (8.1%→2.0%, paired XCOMET
+0.0505 t=+3.87 on 468 source sentences) but *doubles* nulls at 1.5×
(5.2%→11.0%) versus v6.

## Diagnostics already run (generation side looks healthy)

Per-turn stats of the generated audio (identical input turn files per speed):

| run | turns | zero-frame turns | total audio | sec/char median |
|---|---|---|---|---|
| v6 @1.5× | 1183 | 1 (0.1%) | 3366 s | 0.216 |
| v7 @1.5× | 1183 | **0 (0.0%)** | 3256 s | 0.215 |
| v6 @1× | 1743 | 10 (0.6%) | 3267 s | 0.215 |
| v7 @1× | 1743 | 0 (0.0%) | 3215 s | 0.213 |

So at 1.5× v7 speaks the same amount at the same rate with no swallowed turns.
The doubled nulls arise downstream (ASR/alignment) or are noise. A
generation-side variance experiment (regenerate both models once more at 1.5×,
rescore) is running; each cell above is a single sample and gen-side variance
has never been measured (scoring-side ASR is deterministic).

Known instrument caveats (measured previously): Qwen3-ASR transcribes ~22–25%
less text than gpt-4o-mini-transcribe overall and fails on fast/dense speech;
under the OLD gpt-ASR op, null counts on identical audio varied 4/15/18 across
scoring repeats. Under the current Qwen op the scoring is deterministic given
audio, but generation samples at temperature 0.8 with no seed.

## Hypotheses we'd like you to weigh (and find alternatives to)

H1. Single-sample noise: 15/291 vs 35/317 from one generation each.
H2. Distribution shift: the new training rows are 1×-style ultra-short turns
    (0.96 s chunking); 1.5× turn streams have much longer, denser turns that
    the new data does not cover, and the added short-turn bias subtly changes
    prosody/segment boundaries at 1.5× in a way that hurts SEGALE alignment
    granularity (nulls) without hurting the audio itself.
H3. A real pipeline bug somewhere below.

## Pipeline code (complete for the v7 delta)

### 1. New-data converter (trajectory TSV → row requests)

```python
#!/usr/bin/env python3
"""Convert a trajectory-aligned TSV (lab GigaSpeech/FLORAS pipeline) into MOSS
row requests consumable by generate_moss_realtime_long_targets.py.

台账 4.-19。输入是 lab 数据管线 stage-7 的产物：每行一个英文 utterance，
`trajectory` 列是按 0.96s chunk 对齐的中文增量（空串 = 等待）。已验证
6385/6387 行的 trajectory 拼接与 tgt_text 完全一致，因此可以整段合成后按
turn 边界强制对齐切片——与既有管线的约定完全相同：

- 非空增量即 turn；纯标点 turn 无法强制对齐，并入前一个 turn（行首标点并入
  后一个），复用 build_moss_v2_row_requests 的约定；
- 超过 --max-chars-per-call 的行按句末优先切成合成分组；
- turn 中位仅 4 字（0.96s chunk 比 InfiniSST 的 1.92s 细一倍）——短 turn
  正是级联的历史弱点（滑窗吞短句，台账 4.-6），这批数据是有针对性的补充。

usage:
  build_moss_rows_from_trajectory.py --tsv train_..._align.tsv \
      --output-jsonl rows_traj.jsonl [--id-prefix traj] [--max-rows 0]
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import unicodedata
from pathlib import Path

SENTENCE_FINAL = "。！？!?…"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--id-prefix", default="traj")
    parser.add_argument("--max-chars-per-call", type=int, default=300)
    parser.add_argument("--max-duration-s", type=float, default=120.0,
                        help="drop utterances longer than this (runaway 训练风险，"
                             "沿用 30s 规则的精神但放宽——本数据集中位 38.4s)")
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def spoken_chars(text: str) -> int:
    return sum(1 for ch in text if unicodedata.category(ch)[0] in ("L", "N"))


def main() -> None:
    args = parse_args()
    csv.field_size_limit(10**8)
    stats = {"rows_in": 0, "dirty": 0, "too_long": 0, "no_turns": 0,
             "rows_out": 0, "turns_out": 0, "punct_merged": 0}

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.tsv.open(newline="") as handle, args.output_jsonl.open("w") as out:
        for row in csv.DictReader(handle, delimiter="\t"):
            stats["rows_in"] += 1
            raw = (row.get("trajectory") or "").lstrip()
            if not raw.startswith("["):
                stats["dirty"] += 1
                continue
            try:
                traj = ast.literal_eval(raw)
                assert isinstance(traj, list)
            except (ValueError, SyntaxError, AssertionError):
                stats["dirty"] += 1
                continue
            if int(row["n_frames"]) / 16000 > args.max_duration_s:
                stats["too_long"] += 1
                continue

            # 非空增量 -> turn；纯标点 turn 并入邻居（同 v2 约定）
            deltas = [t for t in traj if t.strip()]
            merged: list[str] = []
            for d in deltas:
                if merged and spoken_chars(d) == 0:
                    merged[-1] += d
                    stats["punct_merged"] += 1
                elif not merged and spoken_chars(d) == 0 and len(deltas) > 1:
                    # 行首纯标点：挂到占位符，与下一个增量合并
                    merged.append(d)
                    stats["punct_merged"] += 1
                else:
                    if merged and spoken_chars(merged[-1]) == 0:
                        merged[-1] += d
                    else:
                        merged.append(d)
            merged = [m for m in merged if spoken_chars(m) > 0]
            if not merged:
                stats["no_turns"] += 1
                continue

            row_id = f"{args.id_prefix}_{row['id']}"
            segments = [{"id": f"{row_id}_t{i:03d}", "text": t} for i, t in enumerate(merged)]

            # 句末优先的合成分组（同 build_moss_v2_row_requests）
            groups: list[list[int]] = []
            current: list[int] = []
            current_chars = 0
            for idx, seg in enumerate(segments):
                seg_chars = len(seg["text"])
                if current and current_chars + seg_chars > args.max_chars_per_call:
                    groups.append(current)
                    current, current_chars = [], 0
                current.append(idx)
                current_chars += seg_chars
                if (current_chars > args.max_chars_per_call * 0.7
                        and seg["text"].rstrip()
                        and seg["text"].rstrip()[-1] in SENTENCE_FINAL):
                    groups.append(current)
                    current, current_chars = [], 0
            if current:
                groups.append(current)

            record = {
                "row_id": row_id,
                "split": "train",
                "segments": segments,
                "groups": groups,
                "full_text": "".join(s["text"] for s in segments),
                "spoken_chars": sum(spoken_chars(s["text"]) for s in segments),
                "source": "lab_trajectory_tsv",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["rows_out"] += 1
            stats["turns_out"] += len(segments)
            if args.max_rows and stats["rows_out"] >= args.max_rows:
                break

    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

```

### 2. Whole-passage synthesis of training targets (batched, in-process)

```python
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

```

### 3. Forced alignment + turn slicing (shared with v6 data; unchanged)

```python
#!/usr/bin/env python3
"""Force-align v2 row wavs and slice codec codes into multi-turn SFT records.

For each generated row wav:
  1. run CTC forced alignment (zh wav2vec2) of the concatenated segment text,
  2. derive segment boundaries (midpoint between adjacent aligned spans),
  3. encode the full wav once with MOSS-Audio-Tokenizer,
  4. slice the [T, NQ] codes at boundary frames,
  5. emit one prepared record per row with N assistant turns (text + codes),
     ready for moss_tts_realtime/finetuning/sft.py (prepare_data.py is skipped).

# note (luojiaxuan): chars absent from the aligner vocab (digits, latin, rare
# hanzi) are skipped; a segment with no aligned chars gets proportional
# boundaries interpolated by char count. Per-row alignment coverage and scores
# go to the audit JSONL; rows below --min-coverage are excluded from training.
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import torch
import torchaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-raw-jsonl", required=True, help="accepted rows from generate_moss_realtime_long_targets.py")
    parser.add_argument("--rows-jsonl", required=True, help="row requests from build_moss_v2_row_requests.py")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--codec-path", default="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    parser.add_argument("--align-model", default="jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
    parser.add_argument("--fixed-ref", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def is_spoken(ch: str) -> bool:
    return unicodedata.category(ch)[0] in {"L", "N"}


def load_mono(path: str, target_sr: int, device: torch.device) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    return wav.to(device)


@torch.no_grad()
def encode_codes(codec, wav: torch.Tensor) -> torch.Tensor:
    """Encode a mono [1, S] waveform to [T, NQ] int64 codes (mirrors prepare_data.py)."""
    enc = codec.batch_encode([wav.squeeze(0)], num_quantizers=None)
    codes = enc.audio_codes  # [NQ, B, T]
    length = int(enc.audio_codes_lengths[0].item())
    return codes[:, 0, :length].transpose(0, 1).cpu()


@torch.no_grad()
def align_row(
    aligner, vocab: dict[str, int], blank_id: int, wav16: torch.Tensor, segments: list[dict]
) -> dict[str, Any]:
    """Return segment spans (seconds) plus alignment QA for one row."""
    tokens: list[int] = []
    token_seg: list[int] = []
    total_spoken = 0
    for seg_idx, seg in enumerate(segments):
        for ch in seg["text"]:
            if not is_spoken(ch):
                continue
            total_spoken += 1
            tok = vocab.get(ch) or vocab.get(ch.upper()) or vocab.get(ch.lower())
            if tok is None or tok == blank_id:
                continue
            tokens.append(tok)
            token_seg.append(seg_idx)
    duration_s = wav16.shape[-1] / 16000.0
    if not tokens:
        return {"coverage": 0.0, "duration_s": duration_s, "spans": None, "mean_score": 0.0}

    logits = aligner(wav16).logits  # [1, T, C]
    log_probs = torch.log_softmax(logits, dim=-1)
    targets = torch.tensor([tokens], dtype=torch.int32, device=log_probs.device)
    frame_labels, frame_scores = torchaudio.functional.forced_align(
        log_probs, targets, blank=blank_id
    )
    spans = torchaudio.functional.merge_tokens(frame_labels[0], frame_scores[0], blank=blank_id)
    sec_per_frame = duration_s / log_probs.shape[1]

    seg_start: dict[int, float] = {}
    seg_end: dict[int, float] = {}
    scores: list[float] = []
    for token_idx, span in enumerate(spans):
        seg_idx = token_seg[token_idx]
        start_s = span.start * sec_per_frame
        end_s = span.end * sec_per_frame
# ... (168 more lines truncated)
```

### 4. Mid-passage-start copies + merge (shared with v6; used for v7 merge)

```python
#!/usr/bin/env python3
"""Build the v6 training set: rows that start mid-passage (no silence anchor).

诊断链见实验台账 4.-6 / 4.-7 / 4.-8。要点：

- 滑窗相对 reset 掉约 3.4 BLEU，损失集中在**短 turn 被吞进上一句**；
- 推理侧重生成修不好（系统性行为，不是采样偶发）；
- "短 turn 上下文罕见"被分布统计否定（训练里反而更多）；
- 真正的分布缺口是**静音锚点**：`align_slice` 让每行都从整段合成的第 0 帧
  起（`frame_cuts[0] = 0`），所以**每个训练上下文都含一个"从静音开始"的
  turn**；而滑窗推理的窗口里几乎永远不含。这也解释了 4.1 的核心发现——
  同样的平均历史长度下，会周期性清零的 reset 比从不清零的滑窗高 8 分。

做法：对一部分行，丢掉开头 k 个 turn，让首 turn 的音频从段落中间开始。
不需要重新合成，纯粹是换切片起点，因此几乎零成本。监督保持干净
（不打 ``context_only``），只改变上下文的起始形态。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-jsonl", nargs="+", required=True,
                        help="clean multi-turn records to derive mid-passage copies from")
    parser.add_argument("--base-jsonl", nargs="*", default=[],
                        help="dataset to prepend unchanged (e.g. train_v5.jsonl); "
                             "omit to emit only the mid-passage copies")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--fraction", type=float, default=0.5,
                        help="fraction of eligible clean rows that get a mid-passage copy")
    parser.add_argument("--min-remaining-turns", type=int, default=4,
                        help="a mid-passage copy must keep at least this many turns")
    parser.add_argument("--max-drop-frac", type=float, default=0.5,
                        help="drop at most this fraction of a row's turns from the front")
    parser.add_argument("--id-suffix", default="_v6mid")
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"base_rows": 0, "eligible": 0, "mid_rows": 0,
             "skipped_too_short": 0, "dropped_turns_total": 0}

    with out_path.open("w") as sink:
        for path in args.base_jsonl:
            with open(path) as handle:
                for line in handle:
                    sink.write(line)
                    stats["base_rows"] += 1

        for path in args.clean_jsonl:
            with open(path) as handle:
                for line in handle:
                    row = json.loads(line)
                    turns = row["conversations"]
                    max_drop = int(len(turns) * args.max_drop_frac)
                    if len(turns) - 1 < args.min_remaining_turns or max_drop < 1:
                        stats["skipped_too_short"] += 1
                        continue
                    stats["eligible"] += 1
                    if rng.random() >= args.fraction:
                        continue
                    hi = min(max_drop, len(turns) - args.min_remaining_turns)
                    if hi < 1:
                        stats["skipped_too_short"] += 1
                        continue
                    k = rng.randint(1, hi)
                    out = dict(row)
                    out["id"] = f"{row['id']}{args.id_suffix}{k}"
                    out["conversations"] = [dict(t) for t in turns[k:]]
                    meta = dict(row.get("metadata") or {})
                    # note (luojiaxuan): 记下丢了几个 turn，方便事后核对
                    # "首 turn 是否真的从段落中间开始"这一构造意图。
                    meta["v6_dropped_leading_turns"] = k
                    meta["v6_remaining_turns"] = len(out["conversations"])
                    out["metadata"] = meta
                    sink.write(json.dumps(out, ensure_ascii=False) + "\n")
                    stats["mid_rows"] += 1
                    stats["dropped_turns_total"] += k

    stats["total_rows"] = stats["base_rows"] + stats["mid_rows"]
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

```

### 5. Training launch (v7; v6 identical except 3 procs × accum 5)

```bash
accelerate launch --num_processes 5 --mixed_precision bf16 \
  finetuning/sft.py \
  --model-path OpenMOSS-Team/MOSS-TTS-Realtime \
  --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer \
  --train-jsonl train_v7.jsonl --output-dir ckpt_v7 \
  --per-device-batch-size 1 --gradient-accumulation-steps 3 \
  --learning-rate 1e-5 --num-epochs 1 --num-workers 2 \
  --mixed-precision bf16 --attn-implementation sdpa
```

The trainer is upstream MOSS-TTS `sft.py` plus this patch (the LR-scheduler
part matters: accelerate steps the scheduler num_processes times per optimizer
step; with the patch, process count no longer changes the effective schedule —
v6 used 3 procs, v7 used 5, both under the patch, both global batch 15):

```diff
diff --git a/moss_tts_realtime/finetuning/dataset.py b/moss_tts_realtime/finetuning/dataset.py
index 2008032..d854ff2 100644
--- a/moss_tts_realtime/finetuning/dataset.py
+++ b/moss_tts_realtime/finetuning/dataset.py
@@ -192,7 +192,9 @@ class MossTTSRealtimeSFTDataset(Dataset):
             else:
                 prefill_temp = "<|im_end|>\n<|im_start|>" + role + "\n"
 
-            is_assistant = (role == "assistant")
+            # note (luojiaxuan): context_only turns (repetition-augmented
+            # history) contribute context but no loss.
+            is_assistant = (role == "assistant") and not turn.get("context_only")
             turn_input, turn_label = self._build_turn(
                 text, audio_codes, prefill_temp, is_assistant=is_assistant,
             )
diff --git a/moss_tts_realtime/finetuning/sft.py b/moss_tts_realtime/finetuning/sft.py
index 67800a7..ca9a9eb 100644
--- a/moss_tts_realtime/finetuning/sft.py
+++ b/moss_tts_realtime/finetuning/sft.py
@@ -487,11 +487,16 @@ def main() -> None:
         f"optimizer_steps_per_epoch={update_steps_per_epoch} "
         f"max_train_steps={max_train_steps}"
     )
+    # note (luojiaxuan): accelerator.prepare() 会把 scheduler 包成每个
+    # optimizer step 推进 num_processes 次，所以这里必须把步数乘回来，
+    # 否则调度会提前 num_processes 倍跑完（实测 3 进程时 lr 在约 1/3
+    # 处就归零，后 2/3 训练等于零学习率空转）。
+    _sched_scale = accelerator.num_processes
     lr_scheduler = get_scheduler(
         name=args.lr_scheduler_type,
         optimizer=optimizer,
-        num_warmup_steps=warmup_steps,
-        num_training_steps=max_train_steps,
+        num_warmup_steps=warmup_steps * _sched_scale,
+        num_training_steps=max_train_steps * _sched_scale,
     )
 
     if using_pre_sharded_files:
diff --git a/moss_tts_realtime/mossttsrealtime/modeling_mossttsrealtime_local.py b/moss_tts_realtime/mossttsrealtime/modeling_mossttsrealtime_local.py
index 6d204fe..5028918 100644
--- a/moss_tts_realtime/mossttsrealtime/modeling_mossttsrealtime_local.py
+++ b/moss_tts_realtime/mossttsrealtime/modeling_mossttsrealtime_local.py
@@ -348,9 +348,8 @@ class MossTTSRealtimeLocalTransformer(MossTTSRealtimeLocalTransformerPreTrainedM
 
         causal_mask = create_causal_mask(
             config=self.config,
-            input_embeds=inputs_embeds,
+            inputs_embeds=inputs_embeds,
             attention_mask=attention_mask,
-            cache_position=cache_position,
             past_key_values=past_key_values,
             position_ids=position_ids,
         )

```

### 6. Inference (turn-by-turn TTS, constant sliding window)

```python
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
# ... (367 more lines truncated)
```

Invocation for every eval cell (only ckpt and rows file vary):

```bash
python3 scripts/moss_multiturn_infer.py \
  --model-path <ckpt> --fixed-ref fixed_zh_ref.wav \
  --rows-jsonl talk<N>.<speed>.swrow.jsonl \
  --device cuda --sliding-window 11 --soft-reset-keep 0 \
  --min-runaway-floor-s 15
```

### 7. Scoring chain

```bash
#!/usr/bin/env bash
# note (luojiaxuan): 新 canonical 打分链（用户裁定 2026-08-11，台账 4.-16）：
#   - ASR：自托管 Qwen3-ASR-1.7B（127.0.0.1:47500，plain sglang）；
#   - BLEU：SEGALE 句级，null（under/over-translation）保留为空假设，不剔除；
#   - XCOMET-XL：reference-based（对齐 Open-LiveTranslate 的模式），但 null
#     主动置零（fixed_xcomet_xl_score=0.0，input 构建器既有行为）。
# 与旧 _gptasr 目录并存：qwen 路径的 rundir 无后缀，不覆盖旧 canonical。
# usage: score_chain_refbased.sh <gpu> <tag> <modearg>
set -u
g="$1"; tag="$2"; mode="$3"
BENCH=/data/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench
RUN=acl6060_live_enzh_cascade_moss${tag}_${mode}_chunk192_speed1
RD=$BENCH/rundirs/$RUN
# note (luojiaxuan): hyper01 上的 SEGALE venv 叫 acl6060-segale，路径不同，
# 用 SEG_PY 环境变量覆盖；缺省仍指 hyper00 的 segale_eval2。
SEG_PY="${SEG_PY:-/data/venvs/segale_eval2/bin/python}"
cd /data/S2S_omni

echo "[1/5] Qwen3-ASR"
python3 "$BENCH/score_generic.py" "$tag" "$mode" qwen3 || exit 2
echo "[2/5] SEGALE inputs"
"$SEG_PY" scripts/build_acl6060_segale_inputs.py --run-dir "$RD" || exit 3
echo "[3/5] SEGALE alignment (GPU $g)"
CUDA_VISIBLE_DEVICES="$g" "$SEG_PY" scripts/run_acl6060_segale_alignment.py \
  --run-dir "$RD" --speech-latency-repo /data/speech-to-speech-latency \
  --target-lang zh --device cuda || exit 4
echo "[4/5] BLEU（句级，null 保留）"
"$SEG_PY" scripts/build_acl6060_xcomet_input.py \
  --run-dir "$RD" --output-jsonl "$RD/xcomet_input.jsonl" \
  --summary-json "$RD/bleu_summary.json" --bleu-tokenizer zh || exit 5
echo "[5/5] XCOMET-XL reference-based（null 置零）"
CUDA_VISIBLE_DEVICES="$g" "$SEG_PY" scripts/run_acl6060_xcomet_xl.py \
  --input-jsonl "$RD/xcomet_input.jsonl" --output-jsonl "$RD/xcomet_segments.jsonl" \
  --summary-json "$RD/xcomet_summary.json" --batch-size 4 || exit 6

python3 -c "
import json
b=json.load(open('$RD/bleu_summary.json')); x=json.load(open('$RD/xcomet_summary.json'))
print('NEWOP_RESULT $tag $mode  BLEU %.2f  XCOMET-ref %.4f  null %d/%d (%.1f%%)' % (
  b['bleu'], x['xcomet_xl'], b['null_alignments'], b['segments'], 100*b['null_alignment_ratio']))
"
echo "NEWOP_CHAIN_DONE $tag $mode"

```

Null handling in the BLEU/XCOMET input builder (nulls kept, never dropped):

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

BLEU_TOKENIZER_BY_LANG = {"zh": "zh", "de": "13a", "ja": "ja-mecab"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ACL6060 XCOMET-XL input from SEGALE source-hypothesis alignments."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--aligned-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--bleu-tokenizer", default="")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def read_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_key(run_dir: Path, config: dict[str, Any]) -> str:
    provider = str(config.get("provider") or "")
    target_lang = str(config.get("target_lang") or "")
    chunk_ms = str(config.get("chunk_ms") or "")
    speed = str(config.get("speed_factor") or "")
    return "||".join([run_dir.name, provider, target_lang, chunk_ms, speed])


def null_alignment_type(source: str, hypothesis: str) -> str:
    if not source.strip():
        return "over_translation"
    if not hypothesis.strip():
        return "under_translation"
    return ""


def validate_alignment_coverage(
    segments: list[dict[str, Any]], expected_source_segments: int
) -> None:
    source_ids = [
        int(source_id)
        for segment in segments
        for source_id in list(segment.get("src_ref_ids") or [])
    ]
    if sorted(source_ids) != list(range(1, expected_source_segments + 1)):
        raise ValueError(
            "SEGALE source coverage mismatch: "
            f"expected 1..{expected_source_segments}, got {len(source_ids)} ids"
        )

    target_ids: dict[str, list[int]] = defaultdict(list)
    for segment in segments:
        target_ids[str(segment["doc_id"])].extend(
            int(value) for value in segment.get("mt_indices") or []
        )
    for document, values in target_ids.items():
        if sorted(values) != list(range(len(values))):
            raise ValueError(f"SEGALE target coverage mismatch for {document}: {values}")


def corpus_bleu(rows: list[dict[str, Any]], tokenizer: str) -> float:
    from sacrebleu import corpus_bleu as sacrebleu_corpus_bleu

    return float(
        sacrebleu_corpus_bleu(
            [str(row["hypothesis"]) for row in rows],
            [[str(row["reference"]) for row in rows]],
            tokenize=tokenizer,
        ).score
    )


def build_xcomet_rows(
    run_dir: Path,
    segments: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    key = run_key(run_dir, config)
    rows = []
    for index, segment in enumerate(segments):
        source = str(segment.get("src") or "")
        hypothesis = str(segment.get("tgt") or "")
        reference = str(segment.get("ref") or "")
        null_type = null_alignment_type(source, hypothesis)
        if null_type == "over_translation" and reference.strip():
            raise ValueError(f"over-translation row {index} unexpectedly has a reference")
        rows.append(
            {
                "xcomet_id": f"{key}||segale||{index:04d}",
                "run_key": key,
                "run_dir": str(run_dir),
                "segment_index": index,
                "doc_id": segment.get("doc_id"),
                "segale_segment_id": segment.get("seg_id"),
                "source_segment_ids": list(segment.get("src_ref_ids") or []),
                "hypothesis_sentence_ids": list(segment.get("mt_indices") or []),
                "source": source,
                "hypothesis": hypothesis,
                "reference": reference,
                "null_alignment_type": null_type,
                "fixed_xcomet_xl_score": 0.0 if null_type else None,
                "alignment_backend": "SEGALE",
                "target_lang": config.get("target_lang"),
                "provider": config.get("provider"),
                "chunk_ms": config.get("chunk_ms"),
                "speed_factor": config.get("speed_factor"),
            }
# ... (50 more lines truncated)
```

## Data samples

Converter input row (trajectory column is per-0.96s-chunk zh deltas; empty
string = wait):

```json
{"id": "en00000_10", "n_frames": 337920,
 "trajectory": "['', '', '', '', '', '', '我们', '', '', '', '', '', '', '很快就会在Tdl存储库中发布本次会议的会议纪要和相关材料，', '但', ...]"}
```

Converter output (row request): 6,385 rows, 157,887 turns, concatenation of
turns == tgt_text for 100% of rows; synthesis accepted 6385/6385 with zero
runaway rejections; slicing excluded 0 rows (min coverage 0.5).

Speed-condition input rows (identical for v6 and v7 evals): turn streams from
live InfiniSST runs on 1.25×/1.5× tempo-scaled source audio, 1.92 s chunks;
at 1.5× the turns are longer/denser (1183 turns vs 1743 at 1× for the same
five talks).

## Questions for the auditor

1. Any bug in sections 1–7 that would asymmetrically raise null alignments at
   1.5× for a model trained with extra short-turn data, while *improving* them
   at 1×?
2. Is the sec/char + zero-frame diagnostic sufficient to clear the generation
   side, or is there a failure mode it misses (e.g., prosodic boundary drift
   that hurts sentence alignment without changing totals)?
3. Given one-sample cells, what would you measure next beyond the running
   regenerate-and-rescore experiment?
4. Anything suspicious about mixing an in-process synthesis engine for the new
   training targets with the serving-engine targets of the v6 rows?
