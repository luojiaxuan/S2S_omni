"""Per-piece silence inside a rendered timeline: how much of each turn is speech."""
import json, os, sys, wave, array

ROOT = sys.argv[1]
PEAK = 200            # int16 peak below which a 10 ms frame counts as silent
FRAME_MS = 10

def frames_peak(pcm, sr):
    step = int(sr * FRAME_MS / 1000)
    out = []
    for i in range(0, len(pcm) - step + 1, step):
        window = pcm[i:i + step]
        out.append(max(max(window), -min(window)))
    return out

rows = {}
for line in open(os.path.join(ROOT, "instances.log"), encoding="utf-8"):
    if line.strip():
        r = json.loads(line)
        src = r["source"][0] if isinstance(r["source"], list) else r["source"]
        rows[os.path.splitext(os.path.basename(src))[0]] = r

print(f"{'talk':<20} {'pieces':>6} {'audio_s':>8} {'voiced_s':>9} {'lead_s':>7} {'tail_s':>7} "
      f"{'inner_s':>8} {'lead_ms/pc':>10} {'tail_ms/pc':>10}")
for stem, r in sorted(rows.items()):
    path = os.path.join(ROOT, "wavs", f"{r['index']}_pred.wav")
    with wave.open(path, "rb") as h:
        sr = h.getframerate()
        pcm = array.array("h", h.readframes(h.getnframes()))
    offset = r.get("prediction_offset", 0.0) if "prediction_offset" in r else 0.0
    total = lead = tail = voiced = 0.0
    for start, dur in r["intervals"]:
        a = int((start - offset) / 1000 * sr)
        b = a + int(dur / 1000 * sr)
        peaks = frames_peak(pcm[max(a, 0):b], sr)
        if not peaks:
            continue
        loud = [i for i, p in enumerate(peaks) if p > PEAK]
        total += dur / 1000.0
        if not loud:
            lead += dur / 1000.0
            continue
        lead += loud[0] * FRAME_MS / 1000.0
        tail += (len(peaks) - 1 - loud[-1]) * FRAME_MS / 1000.0
        voiced += sum(1 for p in peaks if p > PEAK) * FRAME_MS / 1000.0
    n = len(r["intervals"])
    inner = total - lead - tail - voiced
    print(f"{stem:<20} {n:6d} {total:8.1f} {voiced:9.1f} {lead:7.1f} {tail:7.1f} {inner:8.1f} "
          f"{1000*lead/n:10.1f} {1000*tail/n:10.1f}")
