"""Ending offset under a tightened runaway budget, with and without the poisoning it prevents."""
import json, os

OLT = "/data04/jaxan/olt_build"
RUN = "/data02/jaxan/tmp/runs/20260830-200824-401864000/eval/output/c2"
FLOOR, SPC = 5.0, 0.4          # budget = max(5 s, chars / 2.5 chars per second)

def spoken(t): return sum(1 for c in t if not c.isspace())
def replay(arr, dur):
    end = 0.0
    for a, d in zip(arr, dur):
        end = max(end, a) + d
    return end

rows = {}
for line in open(f"{OLT}/ours_ext/instances.log", encoding="utf-8"):
    r = json.loads(line)
    rows[os.path.splitext(os.path.basename(r["source"]))[0]] = r

print(f"{'talk':<20} {'src_s':>6} {'实测':>7} {'收紧预算(截断)':>16} {'再算上不再被污染':>18}")
for stem, r in sorted(rows.items()):
    tid = stem.split(".")[-1]
    turns = json.loads(open(f"{RUN}/talk{tid}.summary.jsonl", encoding="utf-8").readline())["turns"]
    src = r["source_length"] / 1000.0
    arr = [d / 1000.0 for d in r["delays"]]
    dur = [d / 1000.0 for _, d in r["intervals"]]
    bad = [d > 0 and spoken(x["text"]) / d < 2.5 for x, d in zip(turns, dur)]

    trunc = [min(d, max(FLOOR, spoken(x["text"]) * SPC)) for x, d in zip(turns, dur)]
    # a turn whose predecessor was bad is a poisoning victim; with detox it would be healthy
    healed = list(trunc)
    for i in range(1, len(turns)):
        if bad[i] and bad[i - 1]:
            healed[i] = spoken(turns[i]["text"]) / 5.0
    print(f"{stem:<20} {src:6.0f} {replay(arr,dur)-src:6.0f}s {replay(arr,trunc)-src:15.0f}s "
          f"{replay(arr,healed)-src:17.0f}s")
