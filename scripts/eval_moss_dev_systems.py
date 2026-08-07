#!/usr/bin/env python3
"""Score dev-set TTS systems (v1 per-segment / v2 multi-turn) for InfiniSST.

Inputs are the synthesis summaries (per-row wav + per-turn durations). For
each system this script computes:
  - duration ratio vs the source row audio (playback backlog proxy),
  - runaway turn rate (turn duration > max(floor, spoken_chars * spc_cap)),
  - whisper large-v3 zh ASR on the row wav, then CER (jiwer) and
    BLEU / chrF (sacrebleu, tokenize=zh) against the concatenated
    InfiniSST segment text (TTS input = reference).
Row-level records go to --output-jsonl; the aggregate goes to --report-json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import unicodedata
import wave
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-jsonl", nargs="+", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--manifest-jsonl", required=True, help="dev_moss_requests.jsonl for source durations")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seconds-per-char", type=float, default=0.6)
    parser.add_argument("--runaway-floor-s", type=float, default=8.0)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def spoken_chars(text: str) -> int:
    return sum(1 for ch in text if unicodedata.category(ch)[0] in {"L", "N"})


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def main() -> None:
    args = parse_args()
    import jiwer
    import sacrebleu
    import whisper

    root = Path(args.dataset_root)
    source_dur: dict[str, float] = defaultdict(float)
    with Path(args.manifest_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            row_id = req["id"].rsplit("_t", 1)[0]
            source_dur[row_id] += wav_duration(root / req["ref_wav"])

    rows = {}
    for path in args.summary_jsonl:
        for line in Path(path).open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.setdefault(str(rec["row_id"]), rec)

    model = whisper.load_model(args.whisper_model, device=args.device)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored = []
    refs, hyps = [], []
    with out_path.open("w", encoding="utf-8") as out:
        for idx, (row_id, rec) in enumerate(sorted(rows.items()), 1):
            ref_text = "".join(t["text"] for t in rec.get("turns", []))
            entry = {
                "row_id": row_id,
                "system": args.system,
                "failure": rec.get("failure"),
                "source_duration_s": round(source_dur.get(row_id, 0.0), 3),
            }
            if rec.get("failure") or not rec.get("wav"):
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                continue
            target_s = float(rec["duration_s"])
            runaway_turns = sum(
                1
                for t in rec["turns"]
                if t["duration_s"]
                > max(args.runaway_floor_s, spoken_chars(t["text"]) * args.max_seconds_per_char)
            )
            asr = model.transcribe(rec["wav"], language="zh", temperature=0.0)
            hyp_text = asr["text"].strip()
            entry.update(
                {
                    "target_duration_s": round(target_s, 3),
                    "duration_ratio": round(target_s / source_dur[row_id], 4)
                    if source_dur.get(row_id)
                    else None,
                    "num_turns": len(rec["turns"]),
                    "runaway_turns": runaway_turns,
                    "ref_text": ref_text,
                    "asr_text": hyp_text,
                    "cer": round(
                        jiwer.cer(" ".join(ref_text), " ".join(hyp_text)) if ref_text else 1.0, 4
                    ),
                }
            )
            refs.append(ref_text)
            hyps.append(hyp_text)
            scored.append(entry)
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if args.log_every > 0 and idx % args.log_every == 0:
                print(json.dumps({"scored": idx, "total": len(rows)}), flush=True)

    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh")
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    ratios = [e["duration_ratio"] for e in scored if e.get("duration_ratio")]
    cers = [e["cer"] for e in scored]
    report = {
        "system": args.system,
        "rows_scored": len(scored),
        "rows_failed": len(rows) - len(scored),
        "bleu_zh": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "cer_mean": round(statistics.mean(cers), 4) if cers else None,
        "cer_median": round(statistics.median(cers), 4) if cers else None,
        "duration_ratio_mean": round(statistics.mean(ratios), 4) if ratios else None,
        "duration_ratio_median": round(statistics.median(ratios), 4) if ratios else None,
        "duration_ratio_p90": round(sorted(ratios)[int(len(ratios) * 0.9)], 4) if ratios else None,
        "total_runaway_turns": sum(e["runaway_turns"] for e in scored),
        "total_turns": sum(e["num_turns"] for e in scored),
    }
    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
