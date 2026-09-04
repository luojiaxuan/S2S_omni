import json, statistics as st, sys
base = "/data04/jaxan/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench/tts_rows"
for tag, name in (("phrase 策略 phrv2e1(打分时实际在用)", "talk110.phrv2e1.swrow.jsonl"),
                  ("固定 chunk 策略 chunk192(对照)", "talk110.chunk192.swrow.jsonl")):
    segs = json.loads(open(f"{base}/{name}").readline())["segments"]
    print(f"=== {tag} — {len(segs)} 段")
    for s in segs[4:12]:
        print(f"  {s['delay_ms']/1000:7.2f}s  {len(s['text']):3d}字  {s['text']}")
    ch = [len(s["text"]) for s in segs]
    gaps = [b["delay_ms"] - a["delay_ms"] for a, b in zip(segs, segs[1:])]
    print(f"  -- 每段字数: 中位 {st.median(ch):.0f} / 均值 {sum(ch)/len(ch):.1f} / 最大 {max(ch)}"
          f"；相邻产出间隔: 中位 {st.median(gaps)/1000:.2f}s / 最小 {min(gaps)/1000:.2f}s")
    print()
