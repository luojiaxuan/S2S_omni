#!/usr/bin/env python3
"""分解 v8 在 1.5× 的掉分：per-talk BLEU、TTS+ASR 保真度、TTS 失控率。

纯 CPU，只读现成评测产物，不重跑合成或打分。
usage: diagnose_v8_speed_drop.py --eval-dir <.../eval/phrase-matched-tts-v8-20260830>
"""
from __future__ import annotations

import argparse
import collections
import json
import unicodedata
from pathlib import Path

TALKS = (110, 117, 268, 367, 590)
CONDS = (("c2", "1x"), ("c3", "1.5x"))
# note (luojiaxuan): 正常发音速率实测 p50 = 0.231 秒/字，两个速度档一致。
# 0.6 取在 p99(≈1.0) 与 p90(≈0.32) 之间，用来圈出「短输入长音频」的失控 turn。
RUNAWAY_SEC_PER_CHAR = 0.6


def alnum(text: str) -> str:
    return "".join(c for c in text if unicodedata.category(c)[0] in ("L", "N"))


def edit_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def per_talk_bleu(eval_dir: Path, cond: str):
    import sacrebleu

    rows = [json.loads(l) for l in
            (eval_dir / "result" / cond / "xcomet_input.jsonl").open(encoding="utf-8")]
    per = collections.defaultdict(lambda: {"h": [], "r": [], "null": 0})
    for r in rows:
        doc = r["doc_id"].replace("2022.acl-long.", "").replace(".wav", "")
        if r.get("null_alignment_type"):
            per[doc]["null"] += 1
        per[doc]["h"].append(r["hypothesis"] or "")
        per[doc]["r"].append(r["reference"] or "")
    out = {d: {"segments": len(v["h"]), "null": v["null"],
               "bleu": round(sacrebleu.corpus_bleu(v["h"], [v["r"]], tokenize="zh").score, 2)}
           for d, v in per.items()}
    allh = [x for v in per.values() for x in v["h"]]
    allr = [x for v in per.values() for x in v["r"]]
    out["ALL"] = {"segments": len(allh), "null": sum(v["null"] for v in per.values()),
                  "bleu": round(sacrebleu.corpus_bleu(allh, [allr], tokenize="zh").score, 2)}
    return out


def fidelity_and_runaway(eval_dir: Path, cond: str):
    buckets = collections.defaultdict(lambda: [0, 0, 0])   # ref_chars, errors, turns
    cers, sec_per_char, runaway = [], [], collections.Counter()
    total_audio = runaway_audio = 0.0
    for talk in TALKS:
        asr = {}
        for line in (eval_dir / "output" / cond / "bc_asr_cache"
                     / f"talk{talk}.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            asr[r["turn_index"]] = (r.get("text") or "").strip()
        row = json.loads((eval_dir / "output" / cond
                          / f"talk{talk}.summary.jsonl").read_text(encoding="utf-8").strip())
        for i, turn in enumerate(row["turns"]):
            ref, hyp = alnum(turn["text"]), alnum(asr.get(i, ""))
            if not ref:
                continue
            err = edit_distance(ref, hyp)
            cers.append(err / len(ref))
            key = ("1-5" if len(ref) <= 5 else "6-15" if len(ref) <= 15
                   else "16-30" if len(ref) <= 30 else "31+")
            b = buckets[key]
            b[0] += len(ref); b[1] += err; b[2] += 1
            dur = float(turn["duration_s"])
            total_audio += dur
            spc = dur / len(ref)
            sec_per_char.append(spc)
            if spc > RUNAWAY_SEC_PER_CHAR:
                runaway[key] += 1
                runaway_audio += dur
    cers.sort(); sec_per_char.sort()
    q = lambda v, p: round(v[min(int(len(v) * p), len(v) - 1)], 3)
    return {
        "turns": len(cers),
        "cer_overall": round(sum(b[1] for b in buckets.values())
                             / sum(b[0] for b in buckets.values()) * 100, 2),
        "cer_by_length": {k: round(v[1] / v[0] * 100, 2) for k, v in sorted(buckets.items())},
        "turns_by_length": {k: v[2] for k, v in sorted(buckets.items())},
        "cer_p50": q(cers, .5), "cer_p90": q(cers, .9),
        "turns_cer_over_50pct": sum(c > 0.5 for c in cers),
        "sec_per_char_p50": q(sec_per_char, .5),
        "sec_per_char_p90": q(sec_per_char, .9),
        "sec_per_char_p99": q(sec_per_char, .99),
        "runaway_turns": sum(runaway.values()),
        "runaway_by_length": dict(runaway),
        "runaway_audio_s": round(runaway_audio, 1),
        "total_audio_s": round(total_audio, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, type=Path)
    ap.add_argument("--out-json", type=Path)
    args = ap.parse_args()
    report = {"runaway_threshold_sec_per_char": RUNAWAY_SEC_PER_CHAR, "conditions": {}}
    for cond, name in CONDS:
        report["conditions"][name] = {
            "bleu": per_talk_bleu(args.eval_dir, cond),
            "tts_asr": fidelity_and_runaway(args.eval_dir, cond),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                 encoding="utf-8")


if __name__ == "__main__":
    main()
