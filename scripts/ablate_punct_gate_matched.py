#!/usr/bin/env python3
"""配对消融：控制 write 长度后，「以标点结尾」还是否影响 TTS 保真度。

外审指出 25.8% 强制切断的统计是同义反复（规则本就规定超时即违反标点偏好），
不能证伪标点前提；缺的是同长度下的配对比较。现有 v8 评测数据里两类 write
天然共存，故可零成本比较，无需重跑 TTS。

两个因变量分开报（外审要求把插入与替换/删除分开）：
- 字错率：编辑距离 / 参考字数，混合了替换、删除、插入；
- 超额音频：实际时长 − 字数 × 0.231（实测 p50 秒/字），只反映「多说了」。

usage: ablate_punct_gate_matched.py --eval-dir <.../phrase-matched-tts-v8-20260830>
"""
from __future__ import annotations

import argparse
import json
import statistics
import unicodedata
from pathlib import Path

TALKS = (110, 117, 268, 367, 590)
CONDS = (("c2", "1x"), ("c3", "1.5x"))
PHRASE_PUNCT = "。！？!?…，,、；;：:"
SEC_PER_CHAR = 0.231
BUCKETS = ((1, 5), (6, 15), (16, 30), (31, 10 ** 6))


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


def bucket_of(n: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"
    raise ValueError(n)


def collect(eval_dir: Path, cond: str):
    rows = []
    for talk in TALKS:
        asr = {}
        for line in (eval_dir / "output" / cond / "bc_asr_cache"
                     / f"talk{talk}.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            asr[r["turn_index"]] = (r.get("text") or "").strip()
        summary = json.loads((eval_dir / "output" / cond
                              / f"talk{talk}.summary.jsonl").read_text(encoding="utf-8").strip())
        for i, turn in enumerate(summary["turns"]):
            text = turn["text"].strip()
            ref = alnum(text)
            if not ref:
                continue
            rows.append({
                "n": len(ref),
                "ends_punct": text[-1] in PHRASE_PUNCT,
                "cer": edit_distance(ref, alnum(asr.get(i, ""))) / len(ref),
                "excess_s": float(turn["duration_s"]) - len(ref) * SEC_PER_CHAR,
            })
    return rows


def summarise(group):
    if not group:
        return None
    excess = sorted(r["excess_s"] for r in group)
    return {
        "turns": len(group),
        "median_chars": statistics.median(r["n"] for r in group),
        "cer_mean_pct": round(statistics.fmean(r["cer"] for r in group) * 100, 2),
        "cer_median_pct": round(statistics.median(r["cer"] for r in group) * 100, 2),
        "excess_over_3s_pct": round(sum(e > 3.0 for e in excess) / len(excess) * 100, 2),
        "excess_p90_s": round(excess[min(int(len(excess) * .9), len(excess) - 1)], 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, type=Path)
    ap.add_argument("--out-json", type=Path)
    args = ap.parse_args()

    report = {"sec_per_char_baseline": SEC_PER_CHAR, "conditions": {}}
    for cond, name in CONDS:
        rows = collect(args.eval_dir, cond)
        per_bucket = {}
        for lo, hi in BUCKETS:
            key = f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"
            sel = [r for r in rows if lo <= r["n"] <= hi]
            per_bucket[key] = {
                "ends_punct": summarise([r for r in sel if r["ends_punct"]]),
                "no_punct": summarise([r for r in sel if not r["ends_punct"]]),
            }
        report["conditions"][name] = {
            "overall": {"ends_punct": summarise([r for r in rows if r["ends_punct"]]),
                        "no_punct": summarise([r for r in rows if not r["ends_punct"]])},
            "by_length": per_bucket,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                 encoding="utf-8")


if __name__ == "__main__":
    main()
