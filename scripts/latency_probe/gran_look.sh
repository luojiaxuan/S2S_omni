#!/usr/bin/env bash
export PYTHONPATH=/data/tmp/runs/20260830-200824-401864000/env/site
grep -h EXIT_gran /data/delta_tts/*.log
python - <<'PY'
import glob, json, statistics as st
SRC = {"110": 703.0, "117": 729.0, "268": 737.4}
SW = "/d4/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench/tts_rows"
def spoken(t): return sum(1 for c in t if not c.isspace())
rows = {}
for p in sorted(glob.glob("/data/delta_tts/gran.s*.jsonl")):
    for line in open(p, encoding="utf-8"):
        if line.strip():
            d = json.loads(line); rows[d["row_id"]] = d
print(f"{'talk':>5} {'turns':>6} {'audio_s':>8} {'aud/src':>8} {'坏turn':>7} {'报警':>5} "
      f"{'收尾偏移':>9} {'中位字/秒':>10} {'中位turn字数':>12}")
for rid, d in sorted(rows.items()):
    tid = rid.split("_")[0].replace("talk", "")
    segs = json.loads(open(f"{SW}/talk{tid}.chunk192.swrow.jsonl", encoding="utf-8").readline())["segments"]
    t = d["turns"]
    end = 0.0
    for s, x in zip(segs, t):
        end = max(end, s["delay_ms"] / 1000.0) + x["duration_s"]
    rate = [spoken(x["text"]) / x["duration_s"] for x in t if x["duration_s"] > 0]
    bad = sum(1 for x in t if x["duration_s"] > 0 and spoken(x["text"]) / x["duration_s"] < 2.5)
    flag = sum(1 for x in t if x.get("runaway_skipped"))
    print(f"{tid:>5} {len(t):6d} {d['duration_s']:8.1f} {d['duration_s']/SRC[tid]:8.2f} {bad:7d} {flag:5d} "
          f"{end - SRC[tid]:8.0f}s {st.median(rate):10.2f} {st.median([spoken(x['text']) for x in t]):12.0f}")
PY
