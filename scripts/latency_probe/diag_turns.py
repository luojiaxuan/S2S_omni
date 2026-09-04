import json, os, statistics as st
RUN = "/data02/jaxan/tmp/runs/20260830-200824-401864000/eval/output/c2"
for tid in ("110", "117", "268"):
    d = json.loads(open(f"{RUN}/talk{tid}.summary.jsonl", encoding="utf-8").readline())
    t = d["turns"]
    print(f"=== talk{tid}: {len(t)} turns, {d['duration_s']:.1f}s, codec_context={d['codec_context']}, "
          f"failure={d['failure']}, num_segments={d.get('num_segments')}")
    fr = [x["frames"] for x in t]
    sec_per_frame = [x["duration_s"] / x["frames"] for x in t if x["frames"]]
    rate = [len(x["text"]) / x["duration_s"] for x in t if x["duration_s"] > 0]
    flagged = [i for i, x in enumerate(t) if x.get("runaway_skipped")]
    slow = [i for i, x in enumerate(t) if x["duration_s"] > 0 and len(x["text"]) / x["duration_s"] < 2.5]
    print(f"  frames: median {st.median(fr):.0f}  max {max(fr)}  "
          f"count at max {sum(1 for f in fr if f == max(fr))}  "
          f"top5 {sorted(fr)[-5:]}")
    print(f"  s/frame: median {st.median(sec_per_frame):.4f} (=> {1/st.median(sec_per_frame):.2f} frames/s)")
    print(f"  runaway_skipped=True: {len(flagged)}  {flagged[:20]}")
    print(f"  <2.5 chars/s      : {len(slow)}  {slow[:20]}")
    print(f"  overlap           : {len(set(flagged) & set(slow))} of {len(slow)}")
    print(f"  frames/char: median {st.median([x['frames']/len(x['text']) for x in t if x['text']]):.2f}")
    print()
