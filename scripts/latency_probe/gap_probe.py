"""Where the silence sits inside a rendered timeline: run-length histogram per piece."""
import json, os, sys, wave, array

ROOT = sys.argv[1]
PEAK, FRAME_MS = 200, 10
BUCKETS = [(0, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 99)]

rows = {}
for line in open(os.path.join(ROOT, "instances.log"), encoding="utf-8"):
    if line.strip():
        r = json.loads(line)
        src = r["source"][0] if isinstance(r["source"], list) else r["source"]
        rows[os.path.splitext(os.path.basename(src))[0]] = r

hdr = "  ".join(f"{a:g}-{b:g}s" for a, b in BUCKETS)
print(f"{'talk':<20} {'audio_s':>8} {'silence_s':>10}   {hdr}   {'trim>0.3s':>10}")
for stem, r in sorted(rows.items()):
    with wave.open(os.path.join(ROOT, "wavs", f"{r['index']}_pred.wav"), "rb") as h:
        sr = h.getframerate()
        pcm = array.array("h", h.readframes(h.getnframes()))
    offset = r.get("prediction_offset", 0.0)
    step = int(sr * FRAME_MS / 1000)
    totals = [0.0] * len(BUCKETS)
    audio = silence = 0.0
    for start, dur in r["intervals"]:
        a = int((start - offset) / 1000 * sr)
        b = a + int(dur / 1000 * sr)
        seg = pcm[max(a, 0):b]
        run = 0
        audio += dur / 1000.0
        for i in range(0, len(seg) - step + 1, step):
            w = seg[i:i + step]
            if max(max(w), -min(w)) <= PEAK:
                run += 1
                continue
            if run:
                length = run * FRAME_MS / 1000.0
                silence += length
                for k, (lo, hi) in enumerate(BUCKETS):
                    if lo <= length < hi:
                        totals[k] += length
                run = 0
        if run:
            length = run * FRAME_MS / 1000.0
            silence += length
            for k, (lo, hi) in enumerate(BUCKETS):
                if lo <= length < hi:
                    totals[k] += length
    # what a 0.3 s cap on every silence run would recover
    trim = sum(t * (1 - 0.3 * 2 / (lo + hi)) for t, (lo, hi) in zip(totals, BUCKETS) if lo >= 0.3)
    cells = "  ".join(f"{t:6.0f}s" for t in totals)
    print(f"{stem:<20} {audio:8.1f} {silence:10.1f}   {cells}   {trim:9.0f}s")
