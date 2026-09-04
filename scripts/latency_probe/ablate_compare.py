"""Compare the two ablation arms: bad-turn count, audio total, and the FIFO ending offset."""
import glob, json, os

W = "/data/tts_guard_ablation"
SW = "/d4/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench/tts_rows"

def spoken(t): return sum(1 for c in t if not c.isspace())

def arm(name):
    rows = {}
    for path in sorted(glob.glob(f"{W}/{name}.s*.jsonl")):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                rows[d["row_id"]] = d
    return rows

def delays(tid):
    segs = json.loads(open(f"{SW}/talk{tid}.phrv2e1.swrow.jsonl", encoding="utf-8").readline())["segments"]
    return [s["delay_ms"] / 1000.0 for s in segs]

SRC = {"110": 703.0, "117": 729.0}

print(f"{'arm':<7} {'talk':>5} {'turns':>6} {'audio_s':>8} {'坏turn':>7} {'报警':>5} "
      f"{'坏turn音频':>10} {'收尾偏移':>9} {'中位字/秒':>10}")
import statistics as st
for name in ("base", "tight"):
    rows = arm(name)
    if not rows:
        print(f"{name}: no output yet")
        continue
    for rid, d in sorted(rows.items()):
        tid = rid.split("_")[0].replace("talk", "")
        t = d["turns"]
        arr = delays(tid)[:len(t)]
        bad = [x for x in t if x["duration_s"] > 0 and spoken(x["text"]) / x["duration_s"] < 2.5]
        flag = sum(1 for x in t if x.get("runaway_skipped"))
        end = 0.0
        for a, x in zip(arr, t):
            end = max(end, a) + x["duration_s"]
        rate = [spoken(x["text"]) / x["duration_s"] for x in t if x["duration_s"] > 0]
        print(f"{name:<7} {tid:>5} {len(t):6d} {d['duration_s']:8.1f} {len(bad):7d} {flag:5d} "
              f"{sum(x['duration_s'] for x in bad):9.0f}s {end - SRC[tid]:8.0f}s {st.median(rate):10.2f}")
