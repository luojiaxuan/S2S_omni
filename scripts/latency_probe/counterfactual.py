"""If only the degenerate turns were fixed, what would the ending offset be?"""
import json, os

OLT = "/data04/jaxan/olt_build"
ROWS = "/data04/jaxan/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench/tts_rows"

def replay(arr, dur):
    end, waits = 0.0, []
    for a, d in zip(arr, dur):
        start = max(end, a)
        waits.append(start - a)
        end = start + d
    return end, sum(waits) / len(waits)

rows = {}
for line in open(f"{OLT}/ours_ext/instances.log", encoding="utf-8"):
    r = json.loads(line)
    rows[os.path.splitext(os.path.basename(r["source"]))[0]] = r

print(f"{'talk':<20} {'src_s':>7} {'实测收尾':>9} {'只修坏 turn':>12} {'再+静音裁剪':>13} {'坏 turn 位置(段号)'}")
for stem, r in sorted(rows.items()):
    tid = stem.split(".")[-1]
    segs = json.loads(open(f"{ROWS}/talk{tid}.phrv2e1.swrow.jsonl", encoding="utf-8").readline())["segments"]
    src = r["source_length"] / 1000.0
    arr = [d / 1000.0 for d in r["delays"]]
    dur = [d / 1000.0 for _, d in r["intervals"]]
    bad = [i for i, (s, d) in enumerate(zip(segs, dur))
           if d > 0 and len(s["text"]) / d < 2.5]
    fixed = [len(segs[i]["text"]) / 5.0 if i in set(bad) else d for i, d in enumerate(dur)]
    end0, _ = replay(arr, dur)
    end1, _ = replay(arr, fixed)
    end2, _ = replay(arr, [d * 0.97 for d in fixed])      # a further 3% from trimming long pauses
    runs, cur = [], []
    for i in bad:
        if cur and i == cur[-1] + 1: cur.append(i)
        else:
            if cur: runs.append(cur)
            cur = [i]
    if cur: runs.append(cur)
    shape = " ".join(f"{c[0]}-{c[-1]}" if len(c) > 1 else str(c[0]) for c in runs)
    print(f"{stem:<20} {src:7.0f} {end0-src:8.0f}s {end1-src:11.0f}s {end2-src:12.0f}s  {shape}")
    print(f"{'':<20} {'':>7} {'':>9} {'':>12} {'':>13}  连续段占坏 turn 的 "
          f"{100*sum(len(c) for c in runs if len(c) > 1)/max(len(bad),1):.0f}%")
