#!/usr/bin/env python3
"""Place the cascade's concatenated turn audio on the source clock with zero-jitter FIFO
playout (turn k starts at max(previous end, its InfiniSST emission time)), the same
rule SimulEval's speech-output renderer and build_cascade_speech_timing.py use.

  render_cascade.py --wav talk_full.wav --summary talk.summary.jsonl \
      --swrow talk.swrow.jsonl --out talk_timeline.wav
"""
import argparse
import json

import numpy as np
import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--swrow", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    x, sr = sf.read(a.wav, always_2d=True)
    x = x.mean(axis=1).astype(np.float32)
    turns = json.loads(open(a.summary).readline())["turns"]
    segs = json.loads(open(a.swrow).readline())["segments"]
    assert len(turns) == len(segs), (len(turns), len(segs))
    assert sum(t["samples"] for t in turns) == len(x), (sum(t["samples"] for t in turns), len(x))

    out = np.zeros(0, dtype=np.float32)
    offset = 0
    prev_end = 0.0
    for turn, seg in zip(turns, segs):
        n = int(turn["samples"])
        piece = x[offset:offset + n]
        offset += n
        start = max(prev_end, float(seg["delay_ms"]) / 1000.0)
        start_i = int(round(start * sr))
        if start_i > len(out):
            out = np.concatenate([out, np.zeros(start_i - len(out), dtype=np.float32)])
        out = np.concatenate([out, piece])
        prev_end = start + n / sr
    sf.write(a.out, out, sr, subtype="PCM_16")
    print(f"wrote {a.out}: {len(out) / sr:.1f}s on the source clock ({len(turns)} turns)")


if __name__ == "__main__":
    main()
