#!/usr/bin/env python3
"""First-cut scoring for the InfiniSST + MOSS v2 cascade on ACL6060.

Per (talk, chunk-size) run: concatenate session wavs into one talk-level
target wav, ASR it (whisper large-v3 zh unless an OpenAI key file is given,
then gpt-4o-mini-transcribe windows), and score BLEU/chrF (sacrebleu,
tokenize=zh) against the talk reference plus duration/backlog stats.

# note (luojiaxuan): corpus BLEU over full-talk text is an approximation of
# the canonical SEGALE-aligned BLEU; label results accordingly. XCOMET-XL and
# LongYAAL run separately via the kit-lecture-translator pipeline.
"""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", required=True)
    parser.add_argument("--target-list", required=True, help="acl_zh target.list (one talk reference per line)")
    parser.add_argument("--source-order", default="268,367,590,110,117", help="talk order of target.list lines")
    parser.add_argument("--talk-wav-dir", required=True, help="dir with 2022.acl-long.<talk>.wav sources")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def concat_run(bench: Path, talk: int, chunk: str) -> tuple[Path, float]:
    rows = [
        json.loads(line)
        for line in (bench / "tts_rows" / f"talk{talk}.chunk{chunk}.rows.jsonl").open(encoding="utf-8")
    ]
    out_path = bench / "tts_wavs" / f"talk{talk}.chunk{chunk}.full.wav"
    summaries = {
        json.loads(line)["row_id"]: json.loads(line)
        for line in (bench / "tts_wavs" / f"talk{talk}.chunk{chunk}.summary.jsonl").open(encoding="utf-8")
    }
    pcm, rate = [], None
    for row in rows:
        rec = summaries.get(row["row_id"])
        if rec is None or rec.get("failure") or not rec.get("wav"):
            continue
        with wave.open(rec["wav"], "rb") as handle:
            rate = rate or handle.getframerate()
            pcm.append(handle.readframes(handle.getnframes()))
    with wave.open(str(out_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        for chunk_pcm in pcm:
            handle.writeframes(chunk_pcm)
    duration = sum(len(c) for c in pcm) / 2 / rate
    return out_path, duration


def main() -> None:
    args = parse_args()
    import sacrebleu
    import whisper

    bench = Path(args.bench_dir)
    talk_order = [int(t) for t in args.source_order.split(",")]
    refs_by_talk = {}
    with Path(args.target_list).open(encoding="utf-8") as handle:
        for talk, line in zip(talk_order, handle):
            refs_by_talk[talk] = line.strip()

    model = whisper.load_model(args.whisper_model, device=args.device)
    results = []
    for chunk in ("096", "192"):
        hyps, refs = [], []
        rows_out = []
        for talk in (110, 117, 268, 367, 590):
            wav_path, target_s = concat_run(bench, talk, chunk)
            with wave.open(str(Path(args.talk_wav_dir) / f"2022.acl-long.{talk}.wav"), "rb") as handle:
                source_s = handle.getnframes() / handle.getframerate()
            asr = model.transcribe(str(wav_path), language="zh", temperature=0.0)
            hyp = asr["text"].strip()
            hyps.append(hyp)
            refs.append(refs_by_talk[talk])
            rows_out.append(
                {
                    "talk": talk,
                    "chunk": chunk,
                    "source_s": round(source_s, 2),
                    "target_s": round(target_s, 2),
                    "duration_ratio": round(target_s / source_s, 4),
                    "hyp_chars": len(hyp),
                }
            )
            print(json.dumps(rows_out[-1]), flush=True)
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh")
        chrf = sacrebleu.corpus_chrf(hyps, [refs])
        results.append(
            {
                "system": f"infinisst_mossv2_chunk{chunk}",
                "asr": f"whisper-{args.whisper_model}",
                "bleu_zh_approx": round(bleu.score, 2),
                "chrf_approx": round(chrf.score, 2),
                "talks": rows_out,
                "duration_ratio_mean": round(
                    sum(r["duration_ratio"] for r in rows_out) / len(rows_out), 4
                ),
            }
        )
    Path(args.output_json).write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
