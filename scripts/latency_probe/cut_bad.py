import json, os, wave
OLT = "/data04/jaxan/olt_build"
ROWS = "/data04/jaxan/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench/tts_rows"
OUT = os.path.expanduser("~/cascade_samples_jaxan")

rows = {}
for line in open(f"{OLT}/ours_ext/instances.log", encoding="utf-8"):
    r = json.loads(line)
    rows[os.path.splitext(os.path.basename(r["source"]))[0]] = r

def cut(src, a_s, b_s, dst):
    with wave.open(src, "rb") as h:
        sr, n = h.getframerate(), h.getnframes()
        h.setpos(max(0, int(a_s * sr)))
        pcm = h.readframes(min(n, int(b_s * sr)) - max(0, int(a_s * sr)))
        p = h.getparams()
    with wave.open(os.path.join(OUT, dst), "wb") as w:
        w.setnchannels(p.nchannels); w.setsampwidth(p.sampwidth); w.setframerate(sr)
        w.writeframes(pcm)

notes = []
# three consecutive degenerate turns in talk 117
r = rows["2022.acl-long.117"]
segs = json.loads(open(f"{ROWS}/talk117.phrv2e1.swrow.jsonl", encoding="utf-8").readline())["segments"]
a = r["intervals"][6][0] / 1000.0
b = (r["intervals"][8][0] + r["intervals"][8][1]) / 1000.0
chars = sum(len(segs[i]["text"]) for i in (6, 7, 8))
cut(f"{OLT}/ours_ext/wavs/{r['index']}_pred.wav", a - 1, b + 1, "D_talk117_ours_three_bad_turns.wav")
notes.append(f"[窗口 D] talk 117 第 6/7/8 段连着坏：播放 {a:.1f}-{b:.1f}s({b-a:.1f}s 音频)只念了 {chars} 字 "
             f"= {chars/(b-a):.2f} 字/秒。文本分别是：\n" +
             "\n".join(f"    第{i}段 {len(segs[i]['text'])}字 {r['intervals'][i][1]/1000:.1f}s：{segs[i]['text']}"
                       for i in (6, 7, 8)))

# a healthy stretch of the same talk, for contrast
i0, i1 = 30, 36
a = r["intervals"][i0][0] / 1000.0
b = (r["intervals"][i1][0] + r["intervals"][i1][1]) / 1000.0
chars = sum(len(segs[i]["text"]) for i in range(i0, i1 + 1))
cut(f"{OLT}/ours_ext/wavs/{r['index']}_pred.wav", a, b, "E_talk117_ours_normal_turns.wav")
notes.append(f"[窗口 E] 同一篇的正常段落 第 {i0}-{i1} 段：{b-a:.1f}s 念了 {chars} 字 = {chars/(b-a):.2f} 字/秒。"
             "与窗口 D 同一个模型、同一个音色,差别只在有没有失控。")

open(os.path.join(OUT, "README.txt"), "a", encoding="utf-8").write("\n\n" + "\n\n".join(notes) + "\n")
print("\n\n".join(notes))
