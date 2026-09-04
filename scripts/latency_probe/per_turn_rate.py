"""Per-turn characters per second: is the cascade slow everywhere, or dead in a few turns?"""
import json, os, statistics as st

OLT = "/data04/jaxan/olt_build"
ROWS = "/data04/jaxan/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench/tts_rows"
rows = {}
for line in open(f"{OLT}/ours_ext/instances.log", encoding="utf-8"):
    r = json.loads(line)
    rows[os.path.splitext(os.path.basename(r["source"]))[0]] = r

print(f"{'talk':<20} {'turns':>6} {'median c/s':>11} {'p10':>6} {'p90':>6}   "
      f"{'turns<2.5c/s':>13} {'their audio_s':>14} {'= % of audio':>13}  {'recovered if 5c/s':>18}")
for stem, r in sorted(rows.items()):
    tid = stem.split(".")[-1]
    segs = json.loads(open(f"{ROWS}/talk{tid}.phrv2e1.swrow.jsonl", encoding="utf-8").readline())["segments"]
    dur = [d / 1000.0 for _, d in r["intervals"]]
    assert len(segs) == len(dur), (len(segs), len(dur))
    rate = [len(s["text"]) / d if d > 0 else float("inf") for s, d in zip(segs, dur)]
    slow = [(len(s["text"]), d) for s, d, q in zip(segs, dur, rate) if q < 2.5]
    slow_audio = sum(d for _, d in slow)
    total = sum(dur)
    # what those turns would take at a healthy 5 chars/s
    recovered = slow_audio - sum(c / 5.0 for c, _ in slow)
    order = sorted(rate)
    print(f"{stem:<20} {len(dur):6d} {st.median(rate):11.2f} {order[len(order)//10]:6.2f} "
          f"{order[9*len(order)//10]:6.2f}   {len(slow):13d} {slow_audio:14.0f} "
          f"{100*slow_audio/total:12.0f}% {recovered:17.0f}s")

print()
print("最慢的十个 turn(全部 talk):")
worst = []
for stem, r in rows.items():
    tid = stem.split(".")[-1]
    segs = json.loads(open(f"{ROWS}/talk{tid}.phrv2e1.swrow.jsonl", encoding="utf-8").readline())["segments"]
    for i, (s, (_, d)) in enumerate(zip(segs, r["intervals"])):
        if d > 0:
            worst.append((len(s["text"]) / (d / 1000.0), stem, i, len(s["text"]), d / 1000.0, s["text"]))
for q, stem, i, n, d, text in sorted(worst)[:10]:
    print(f"  {q:5.2f} 字/秒  {stem} 第{i:3d}段  {n:3d}字 {d:6.2f}s  {text[:34]}")
