"""Is a bad turn predictable from its text or its position?"""
import json, statistics as st

RUN = "/data02/jaxan/tmp/runs/20260830-200824-401864000/eval/output/c2"
PUNC = "。，、？！；：)）」』”…—"

def spoken(t): return sum(1 for c in t if not c.isspace())

rows = []
for tid in ("110", "117", "268"):
    t = json.loads(open(f"{RUN}/talk{tid}.summary.jsonl", encoding="utf-8").readline())["turns"]
    for i, x in enumerate(t):
        d = x["duration_s"]
        rows.append({"talk": tid, "i": i, "n": len(t), "chars": spoken(x["text"]),
                     "bad": d > 0 and spoken(x["text"]) / d < 2.5,
                     "endpunc": x["text"].strip()[-1] in PUNC if x["text"].strip() else False,
                     "startpunc": x["text"].strip()[0] in PUNC if x["text"].strip() else False,
                     "dur": d})

bad = [r for r in rows if r["bad"]]
good = [r for r in rows if not r["bad"]]
print(f"坏 turn {len(bad)} / 共 {len(rows)}")
print(f"  字数        坏: 中位 {st.median([r['chars'] for r in bad]):.0f} "
      f"均值 {sum(r['chars'] for r in bad)/len(bad):.1f} / "
      f"正常: 中位 {st.median([r['chars'] for r in good]):.0f} "
      f"均值 {sum(r['chars'] for r in good)/len(good):.1f}")
print(f"  以标点结尾  坏: {100*sum(r['endpunc'] for r in bad)/len(bad):.0f}% / "
      f"正常: {100*sum(r['endpunc'] for r in good)/len(good):.0f}%")
print(f"  以标点开头  坏: {100*sum(r['startpunc'] for r in bad)/len(bad):.0f}% / "
      f"正常: {100*sum(r['startpunc'] for r in good)/len(good):.0f}%")
print()
print("按 turn 在整篇里的相对位置分五档（坏 turn 占比）:")
for lo in range(0, 100, 20):
    sel = [r for r in rows if lo <= 100 * r["i"] / r["n"] < lo + 20]
    print(f"  {lo:3d}-{lo+20:3d}%  {sum(r['bad'] for r in sel):3d}/{len(sel):3d} = "
          f"{100*sum(r['bad'] for r in sel)/len(sel):4.1f}%")
print()
print("按字数分档（坏 turn 占比）:")
for lo, hi in ((0, 10), (10, 15), (15, 20), (20, 25), (25, 100)):
    sel = [r for r in rows if lo <= r["chars"] < hi]
    if sel:
        print(f"  {lo:2d}-{hi:3d} 字  {sum(r['bad'] for r in sel):3d}/{len(sel):3d} = "
              f"{100*sum(r['bad'] for r in sel)/len(sel):4.1f}%")
