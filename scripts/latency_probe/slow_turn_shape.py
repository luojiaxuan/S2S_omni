"""What is inside the slow turns: silence, or audio that is not the text?"""
import json, os, wave, array

OLT = "/data04/jaxan/olt_build"
ROWS = "/data04/jaxan/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench/tts_rows"
PEAK, FRAME_MS = 200, 10

rows = {}
for line in open(f"{OLT}/ours_ext/instances.log", encoding="utf-8"):
    r = json.loads(line)
    rows[os.path.splitext(os.path.basename(r["source"]))[0]] = r

print(f"{'talk/seg':<28} {'chars':>5} {'dur_s':>6} {'voiced_s':>8} {'voiced%':>8} "
      f"{'lead_s':>7} {'tail_s':>7} {'c/s over voiced':>16}")
agg = {}
for stem, r in sorted(rows.items()):
    tid = stem.split(".")[-1]
    segs = json.loads(open(f"{ROWS}/talk{tid}.phrv2e1.swrow.jsonl", encoding="utf-8").readline())["segments"]
    with wave.open(f"{OLT}/ours_ext/wavs/{r['index']}_pred.wav", "rb") as h:
        sr = h.getframerate()
        pcm = array.array("h", h.readframes(h.getnframes()))
    step = int(sr * FRAME_MS / 1000)
    tot = {"slow": [0.0, 0.0, 0], "rest": [0.0, 0.0, 0]}
    shown = 0
    for i, (s, (start, dur)) in enumerate(zip(segs, r["intervals"])):
        if dur <= 0:
            continue
        rate = len(s["text"]) / (dur / 1000.0)
        a = int(start / 1000 * sr); b = a + int(dur / 1000 * sr)
        seg = pcm[a:b]
        peaks = [max(max(seg[j:j+step]), -min(seg[j:j+step]))
                 for j in range(0, len(seg) - step + 1, step)]
        loud = [j for j, p in enumerate(peaks) if p > PEAK]
        voiced = len(loud) * FRAME_MS / 1000.0
        key = "slow" if rate < 2.5 else "rest"
        tot[key][0] += dur / 1000.0; tot[key][1] += voiced; tot[key][2] += len(s["text"])
        if rate < 2.5 and shown < 3:
            lead = (loud[0] if loud else len(peaks)) * FRAME_MS / 1000.0
            tail = (len(peaks) - 1 - loud[-1]) * FRAME_MS / 1000.0 if loud else 0.0
            print(f"{stem.split('.')[-1]+' 第'+str(i)+'段':<28} {len(s['text']):5d} {dur/1000:6.2f} "
                  f"{voiced:8.2f} {100*voiced/(dur/1000):7.0f}% {lead:7.2f} {tail:7.2f} "
                  f"{len(s['text'])/voiced if voiced else 0:16.2f}")
            shown += 1
    agg[stem] = tot

print()
print(f"{'talk':<20} {'组':<6} {'audio_s':>8} {'voiced_s':>9} {'voiced%':>8} {'chars':>7} {'c/s(总)':>9} {'c/s(发声)':>10}")
for stem, tot in sorted(agg.items()):
    for key, label in (("rest", "正常"), ("slow", "<2.5c/s")):
        d, v, c = tot[key]
        print(f"{stem:<20} {label:<6} {d:8.0f} {v:9.0f} {100*v/d if d else 0:7.0f}% {c:7d} "
              f"{c/d if d else 0:9.2f} {c/v if v else 0:10.2f}")
