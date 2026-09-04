"""Does an undetected bad turn poison the next one? The guard clears the window; nothing else does."""
import json

RUN = "/data02/jaxan/tmp/runs/20260830-200824-401864000/eval/output/c2"
FLOOR_S, SEC_PER_CHAR = 15.0, 0.6

def spoken(text):
    return sum(1 for c in text if not c.isspace())

allt = []
for tid in ("110", "117", "268"):
    t = json.loads(open(f"{RUN}/talk{tid}.summary.jsonl", encoding="utf-8").readline())["turns"]
    allt.append((tid, t))

def bad(x):
    return x["duration_s"] > 0 and spoken(x["text"]) / x["duration_s"] < 2.5

print(f"{'talk':>5} {'turns':>6} {'bad':>4} {'flagged':>8}  "
      f"{'P(bad)':>7} {'P(bad | prev bad, 未报警)':>26} {'P(bad | prev 报警)':>20} {'P(bad | prev 正常)':>20}")
tot = [0, 0, 0, 0, 0, 0, 0, 0]
for tid, t in allt:
    b = [bad(x) for x in t]
    f = [bool(x.get("runaway_skipped")) for x in t]
    n_after_badunflagged = sum(1 for i in range(len(t) - 1) if b[i] and not f[i])
    k_after_badunflagged = sum(1 for i in range(len(t) - 1) if b[i] and not f[i] and b[i + 1])
    n_after_flag = sum(1 for i in range(len(t) - 1) if f[i])
    k_after_flag = sum(1 for i in range(len(t) - 1) if f[i] and b[i + 1])
    n_after_good = sum(1 for i in range(len(t) - 1) if not b[i])
    k_after_good = sum(1 for i in range(len(t) - 1) if not b[i] and b[i + 1])
    for j, v in enumerate((len(t), sum(b), sum(f), n_after_badunflagged, k_after_badunflagged,
                           n_after_flag, k_after_flag, 0)):
        tot[j] += v
    tot[7] += n_after_good
    globals().setdefault("_g", []).append((k_after_good, n_after_good))
    def pc(k, n): return f"{k}/{n} = {100*k/n:.0f}%" if n else "n/a"
    print(f"{tid:>5} {len(t):6d} {sum(b):4d} {sum(f):8d}  {100*sum(b)/len(t):6.0f}% "
          f"{pc(k_after_badunflagged, n_after_badunflagged):>26} {pc(k_after_flag, n_after_flag):>20} "
          f"{pc(k_after_good, n_after_good):>20}")

kg = sum(k for k, _ in _g); ng = sum(n for _, n in _g)
print()
print(f"合计: {tot[1]}/{tot[0]} turn 是坏的 ({100*tot[1]/tot[0]:.0f}%)，其中只有 {tot[2]} 个触发了 runaway 报警")
print(f"  下一个 turn 也坏的概率:")
print(f"    上一个坏且未报警 : {tot[4]}/{tot[3]} = {100*tot[4]/tot[3]:.0f}%")
print(f"    上一个坏且已报警 : {tot[6]}/{tot[5]} = {100*tot[6]/tot[5]:.0f}%" if tot[5] else "    上一个坏且已报警 : n/a")
print(f"    上一个正常       : {kg}/{ng} = {100*kg/ng:.0f}%")

print()
print("如果收紧预算(不改别的),能抓住多少坏 turn、以及被截掉多少音频:")
print(f"{'floor_s':>8} {'s/char':>7} {'触发数':>7} {'覆盖坏 turn':>12} {'误伤正常 turn':>14} {'截掉音频':>10}")
for floor_s, spc in ((15.0, 0.6), (8.0, 0.6), (5.0, 0.4), (3.0, 0.3), (2.0, 0.25), (2.0, 0.2)):
    fire = cover = false_fire = 0
    cut = 0.0
    nbad = 0
    for tid, t in allt:
        for x in t:
            n = spoken(x["text"])
            budget = max(floor_s, n * spc)
            isbad = bad(x)
            nbad += isbad
            if x["duration_s"] > budget:
                fire += 1
                cut += x["duration_s"] - budget
                if isbad: cover += 1
                else: false_fire += 1
    print(f"{floor_s:8.1f} {spc:7.2f} {fire:7d} {cover:>7d}/{nbad:<4d} {false_fire:14d} {cut:9.0f}s")
