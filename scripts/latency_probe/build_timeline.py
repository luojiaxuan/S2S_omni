"""Lay a cascade synthesis on the source clock and write the timeline the OLT scorer reads.

Turn k starts at max(previous end, its InfiniSST emission time) — the zero-jitter FIFO rule.
Inputs: the run's concatenated per-talk wav plus its summary (samples per turn), and the
swrow that produced it (per-segment text and delay_ms).
"""
import argparse, json, os, wave
from pathlib import Path

SR_OUT = 24000
TALKS = {"110": 703.0, "117": 729.0, "268": 737.4}
STEM = {"110": "2022.acl-long.110", "117": "2022.acl-long.117", "268": "2022.acl-long.268"}


def read_wav(path):
    with wave.open(str(path), "rb") as h:
        return h.getframerate(), h.getnchannels(), h.getsampwidth(), h.readframes(h.getnframes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", required=True)
    ap.add_argument("--summary-glob", required=True, help="e.g. /data/delta_tts/delta.s*.jsonl")
    ap.add_argument("--swrow-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-dir", required=True)
    a = ap.parse_args()

    import glob
    summaries = {}
    for path in sorted(glob.glob(a.summary_glob)):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                summaries[d["row_id"]] = d

    out = Path(a.out); (out / "wavs").mkdir(parents=True, exist_ok=True)
    records, transcripts = [], []
    for index, tid in enumerate(sorted(TALKS)):
        rid = f"talk{tid}_phrv2ep1fix_full"
        if rid not in summaries:
            raise SystemExit(f"no summary for {rid}")
        summary = summaries[rid]
        segs = json.loads(open(f"{a.swrow_dir}/talk{tid}.phrv2e1.swrow.jsonl",
                               encoding="utf-8").readline())["segments"]
        turns = summary["turns"]
        if len(turns) != len(segs):
            raise SystemExit(f"{rid}: {len(turns)} turns but {len(segs)} swrow segments")
        sr, ch, sw, pcm = read_wav(Path(a.synth_dir) / f"{rid}.wav")
        if (sr, ch, sw) != (SR_OUT, 1, 2):
            raise SystemExit(f"{rid}: wav is {sr} Hz {ch}ch {sw*8}bit, expected 24 kHz mono pcm16")
        total_samples = sum(int(t["samples"]) for t in turns)
        if total_samples != len(pcm) // 2:
            raise SystemExit(f"{rid}: turns sum to {total_samples} samples but wav has {len(pcm)//2}")

        rendered = bytearray()
        intervals, offset, prev_end = [], 0, 0.0
        for turn, seg in zip(turns, segs):
            n = int(turn["samples"])
            piece = pcm[2 * offset: 2 * (offset + n)]
            offset += n
            start = max(prev_end, float(seg["delay_ms"]) / 1000.0)
            start_i = int(round(start * SR_OUT))
            if start_i > len(rendered) // 2:
                rendered.extend(b"\x00\x00" * (start_i - len(rendered) // 2))
            rendered.extend(piece)
            intervals.append([start * 1000.0, n / SR_OUT * 1000.0])
            prev_end = start + n / SR_OUT

        wav_path = out / "wavs" / f"{index}_pred.wav"
        with wave.open(str(wav_path), "wb") as h:
            h.setnchannels(1); h.setsampwidth(2); h.setframerate(SR_OUT)
            h.writeframes(bytes(rendered))
        source = os.path.join(a.source_dir, f"{STEM[tid]}.wav")
        records.append({"index": index, "source": source, "source_length": TALKS[tid] * 1000.0,
                        "delays": [float(s["delay_ms"]) for s in segs],
                        "durations": [d for _, d in intervals], "intervals": intervals})
        transcripts.append({"source": source, "text": "".join(s["text"] for s in segs)})
        print(f"{rid}: {len(turns)} turns, audio {len(rendered)/2/SR_OUT:.1f}s, "
              f"ends {prev_end:.1f}s (source {TALKS[tid]:.0f}s)")

    with open(out / "instances.log", "w", encoding="utf-8") as h:
        for r in records:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "transcripts.jsonl", "w", encoding="utf-8") as h:
        for r in transcripts:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
