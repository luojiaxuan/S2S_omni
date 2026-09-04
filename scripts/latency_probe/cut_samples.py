"""Cut listening samples out of the two scored renderings and the source talk."""
import json, os, wave

OLT = "/data04/jaxan/olt_build"
ROWS = "/data04/jaxan/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench/tts_rows"
OUT = os.path.expanduser("~/cascade_samples_jaxan")
os.makedirs(OUT, exist_ok=True)

def rows(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            src = r["source"][0] if isinstance(r["source"], list) else r["source"]
            out[os.path.splitext(os.path.basename(src))[0]] = r
    return out

def cut(src_wav, a_s, b_s, dst):
    with wave.open(src_wav, "rb") as h:
        sr, n = h.getframerate(), h.getnframes()
        a = max(0, int(a_s * sr)); b = min(n, int(b_s * sr))
        h.setpos(a)
        pcm = h.readframes(b - a)
        params = h.getparams()
    with wave.open(os.path.join(OUT, dst), "wb") as w:
        w.setnchannels(params.nchannels); w.setsampwidth(params.sampwidth)
        w.setframerate(sr); w.writeframes(pcm)
    return (b - a) / sr

O, T = rows(f"{OLT}/ours_ext/instances.log"), rows(f"{OLT}/theirs_ext/instances.log")
notes = []

# 1) a normal stretch of talk 110
a, b = 26.0, 62.0
notes.append(f"[窗口 A] talk 110 源音频 {a:.0f}-{b:.0f}s，三个文件同一窗口，可直接对听。")
cut(f"{OLT}/acl_root/dev/full_wavs/2022.acl-long.110.wav", a, b, "A_talk110_source_EN.wav")
cut(f"{OLT}/ours_ext/wavs/{O['2022.acl-long.110']['index']}_pred.wav", a, b, "A_talk110_ours_v8.wav")
to = T['2022.acl-long.110'].get("prediction_offset", 0.0) / 1000.0
cut(f"{OLT}/theirs_ext/wavs/{T['2022.acl-long.110']['index']}_pred.wav", a - to, b - to,
    "A_talk110_theirs_simuls2st.wav")

# 2) the longest single turn of talk 110 — the runaway generation
r = O["2022.acl-long.110"]
i = max(range(len(r["intervals"])), key=lambda k: r["intervals"][k][1])
start, dur = (v / 1000.0 for v in r["intervals"][i])
segs = json.loads(open(f"{ROWS}/talk110.phrv2e1.swrow.jsonl", encoding="utf-8").readline())["segments"]
text = segs[i]["text"]
notes.append(f"[窗口 B] talk 110 最长的一个 turn：第 {i} 段，播放 {start:.1f}-{start+dur:.1f}s "
             f"({dur:.1f}s)，文本 {len(text)} 字 = {len(text)/dur:.2f} 字/秒：{text}")
cut(f"{OLT}/ours_ext/wavs/{r['index']}_pred.wav", start - 2, start + dur + 2, "B_talk110_ours_longest_turn.wav")

# 3) talk 117 after the source has ended — the backlog you can hear
r = O["2022.acl-long.117"]
src_end = r["source_length"] / 1000.0
notes.append(f"[窗口 C] talk 117 源音频在 {src_end:.0f}s 结束，我方渲染一直播到 "
             f"{(r['intervals'][-1][0]+r['intervals'][-1][1])/1000:.0f}s。这段是 {src_end:.0f}-{src_end+60:.0f}s，"
             "演讲已经结束，译文还在念。")
cut(f"{OLT}/ours_ext/wavs/{r['index']}_pred.wav", src_end, src_end + 60, "C_talk117_ours_after_talk_ended.wav")
tt = T["2022.acl-long.117"]
notes.append(f"     对照：SimulS2ST 在 talk 117 上 {(tt['intervals'][-1][0]+tt['intervals'][-1][1])/1000:.0f}s 就播完了。")

# the phrase text covering window A, for reading along
lines = [f"  {s['delay_ms']/1000:7.2f}s  {s['text']}" for s in segs
         if a - 6 <= s["delay_ms"] / 1000 <= b]
notes.append("[窗口 A 对应的 InfiniSST 逐段产出]\n" + "\n".join(lines))

open(os.path.join(OUT, "README.txt"), "w", encoding="utf-8").write("\n\n".join(notes) + "\n")
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    print(f"{f:<42} {os.path.getsize(p)/1e6:6.2f} MB")
