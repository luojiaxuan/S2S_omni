#!/usr/bin/env python3
"""Build a listening file: left channel = source speech, right channel = translation,
both on the same clock, optionally cropped to [start, end] seconds.

  make_stereo.py --source src.wav --translation pred.wav --offset-ms 0 \
      --start 0 --end 90 --out excerpt.wav
"""
import argparse
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

OUT_SR = 24000


def load_mono(path, target_sr=OUT_SR):
    x, sr = sf.read(path, always_2d=True)
    x = x.mean(axis=1)
    if sr != target_sr:
        g = gcd(target_sr, sr)
        x = resample_poly(x, target_sr // g, sr // g)
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--translation", required=True)
    ap.add_argument("--offset-ms", type=float, default=0.0,
                    help="translation starts this many ms after t=0 (SimulEval's "
                         "prediction_offset when the rendered wav omits the lead-in)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = load_mono(a.source)
    tr = load_mono(a.translation)
    lead = np.zeros(int(round(a.offset_ms / 1000.0 * OUT_SR)), dtype=np.float32)
    tr = np.concatenate([lead, tr])
    n = max(len(src), len(tr))
    src = np.pad(src, (0, n - len(src)))
    tr = np.pad(tr, (0, n - len(tr)))
    s0 = int(a.start * OUT_SR)
    s1 = int(a.end * OUT_SR) if a.end else n
    stereo = np.stack([src[s0:s1], tr[s0:s1]], axis=1)
    peak = np.abs(stereo).max() or 1.0
    stereo = stereo / peak * 0.9
    sf.write(a.out, stereo, OUT_SR, subtype="PCM_16")
    print(f"wrote {a.out}: {(s1 - s0) / OUT_SR:.1f}s  L=source R=translation")


if __name__ == "__main__":
    main()
