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
        seg_start.setdefault(seg_idx, start_s)
        seg_end[seg_idx] = end_s
        scores.append(float(span.score))

    # note (luojiaxuan): 2026-08-20 起未对齐段不再插值兜底，整行 reject
    # （用户裁定 no-fallback；台账 4.-21）。旧插值实现还有实质 bug：
    # gap_segs 收集的是全行所有缺失段而非当前连续缺口，再全部塞进第一个
    # 缺口的区间——v7 traj 超短 turn 大量触发，产出 0–3 帧坏训练目标，
    # 教会模型吞轮。ChatGPT 外部审计发现，本地证实。
    n = len(segments)
    starts = [seg_start.get(i) for i in range(n)]
    ends = [seg_end.get(i) for i in range(n)]
    unaligned = [i for i in range(n) if starts[i] is None]
    coverage = len(spans) / max(1, total_spoken)
    mean_score = sum(scores) / max(1, len(scores))
    if unaligned:
        return {
            "coverage": round(coverage, 4),
            "duration_s": duration_s,
            "spans": None,
            "mean_score": round(mean_score, 4),
            "error": f"unaligned_segments {len(unaligned)}/{n} idx={unaligned[:8]}",
        }
    return {
        "coverage": round(coverage, 4),
        "duration_s": duration_s,
        "spans": list(zip(starts, ends)),
        "mean_score": round(mean_score, 4),
    }


def boundaries_from_spans(spans: list[tuple[float, float]], duration_s: float) -> list[float]:
    """Cut points between adjacent segments: midpoint of end_k and start_{k+1}."""
    cuts = [0.0]
    for k in range(len(spans) - 1):
        end_k = spans[k][1]
        start_next = spans[k + 1][0]
        cuts.append(max(cuts[-1], (end_k + max(start_next, end_k)) / 2.0))
    cuts.append(duration_s)
    return cuts


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_jsonl = Path(args.output_jsonl)
    audit_jsonl = Path(args.audit_jsonl)

    rows_meta = {str(r["row_id"]): r for r in read_jsonl(args.rows_jsonl)}
    done = {str(r["id"]) for r in read_jsonl(output_jsonl)} if output_jsonl.exists() else set()
    if audit_jsonl.exists():
        done |= {str(r["row_id"]) for r in read_jsonl(audit_jsonl) if r.get("excluded")}

    raw_rows = [
        row
        for idx, row in enumerate(read_jsonl(args.row_raw_jsonl))
        if idx % args.num_shards == args.shard_id and str(row["row_id"]) not in done
    ]

    from transformers import AutoModel, Wav2Vec2ForCTC, Wav2Vec2Processor

    codec = AutoModel.from_pretrained(args.codec_path, trust_remote_code=True).eval().to(device)
    codec_sr = int(getattr(codec.config, "sampling_rate", 24000))
    processor = Wav2Vec2Processor.from_pretrained(args.align_model)
    aligner = Wav2Vec2ForCTC.from_pretrained(args.align_model).eval().to(device)
    vocab = processor.tokenizer.get_vocab()
    blank_id = processor.tokenizer.pad_token_id or 0

    ref_codes = None
    if args.fixed_ref:
        ref_wav = load_mono(args.fixed_ref, codec_sr, device)
        ref_codes = encode_codes(codec, ref_wav).tolist()

    emitted = excluded = 0
    for processed, row in enumerate(raw_rows, 1):
        row_id = str(row["row_id"])
        meta = rows_meta[row_id]
        segments = meta["segments"]

        wav16 = load_mono(row["wav"], 16000, device)
        qa = align_row(aligner, vocab, blank_id, wav16, segments)
        audit = {
            "row_id": row_id,
            "split": row.get("split"),
            "coverage": qa["coverage"],
            "mean_score": qa["mean_score"],
            "duration_s": round(qa["duration_s"], 3),
            "num_segments": len(segments),
        }
        if qa["spans"] is None or qa["coverage"] < args.min_coverage:
            audit["excluded"] = True
            if qa.get("error"):
                audit["error"] = qa["error"]
            excluded += 1
            append_jsonl(audit_jsonl, audit)
            continue

        wav_codec = load_mono(row["wav"], codec_sr, device)
        codes = encode_codes(codec, wav_codec)  # [T, NQ]
        total_frames = codes.shape[0]
        fps = total_frames / qa["duration_s"]
        if not 10.0 <= fps <= 15.0:
            audit["excluded"] = True
            audit["error"] = f"unexpected codec fps {fps:.2f}"
            excluded += 1
            append_jsonl(audit_jsonl, audit)
            continue

        cuts = boundaries_from_spans(qa["spans"], qa["duration_s"])
        frame_cuts = [min(total_frames, max(0, round(c * fps))) for c in cuts]
        frame_cuts[0], frame_cuts[-1] = 0, total_frames
        for k in range(1, len(frame_cuts)):  # every turn keeps at least one frame
            if frame_cuts[k] <= frame_cuts[k - 1]:
                frame_cuts[k] = min(total_frames, frame_cuts[k - 1] + 1)
        # note (luojiaxuan): 上面的前向修复在触到 total_frames 后无法再推进，
        # 尾部 turn 会拿到 0 帧目标（= 教模型立即 EOS）。2026-08-20 起这种
        # 行直接 reject 而不是产出坏目标（用户裁定 no-fallback；台账 4.-21）。
        if any(frame_cuts[k] <= frame_cuts[k - 1] for k in range(1, len(frame_cuts))):
            audit["excluded"] = True
            audit["error"] = "frame_budget_exhausted: zero-frame turn after repair"
            excluded += 1
            append_jsonl(audit_jsonl, audit)
            continue

        conversations = []
        for k, seg in enumerate(segments):
            conversations.append(
                {
                    "role": "assistant",
                    "text": seg["text"],
                    "audio_codes": codes[frame_cuts[k] : frame_cuts[k + 1]].tolist(),
                }
            )
        record: dict[str, Any] = {
            "id": row_id,
            "conversations": conversations,
            "metadata": {
                "split": row.get("split"),
                "src_lang": "en",
                "tgt_lang": "zh",
                "duration_s": round(qa["duration_s"], 3),
                "align_coverage": qa["coverage"],
                "align_mean_score": qa["mean_score"],
                "codec_fps": round(fps, 3),
                "segment_ids": [s["id"] for s in segments],
                "boundaries_s": [round(c, 3) for c in cuts],
                "fixed_ref": args.fixed_ref,
            },
        }
        if ref_codes is not None:
            record["ref_audio_codes"] = ref_codes
        append_jsonl(output_jsonl, record)
        append_jsonl(audit_jsonl, audit)
        emitted += 1
        if args.log_every > 0 and processed % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "processed": processed,
                        "emitted": emitted,
                        "excluded": excluded,
                        "remaining": len(raw_rows) - processed,
                    }
                ),
                flush=True,
            )

    print(
        json.dumps(
            {"shard": args.shard_id, "selected": len(raw_rows), "emitted": emitted, "excluded": excluded},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
