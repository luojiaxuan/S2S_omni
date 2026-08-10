"""把平均 XCOMET 拆成「漏译占比」与「对上的句子本身好不好」两项。

# note (luojiaxuan): 配套的选择效应警告见台账 6.8——不要用「非空段均分」
# 跨 run 比较，那个分母不是同一批句子。

# note (luojiaxuan): 两个指标都把空对齐记 0 分，所以
#   mean = (1 - null_ratio) * mean_on_nonnull
# 拆开看才知道滑窗输在哪：是整句被吞掉（null 多），还是对上的句子
# 内容也变差（非空段均分低）。这两种失败要用完全不同的办法修。
"""
import json
import statistics
import sys
from pathlib import Path

BENCH = Path("/data/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench/rundirs")
print("%-26s %7s %8s %9s %10s" % ("run", "总均分", "漏译率", "非空均分", "段数"))
for name, run in (l.split("=", 1) for l in sys.argv[1:]):
    rd = BENCH / run
    try:
        scores, nulls = [], 0
        for line in (rd / "xcomet_segments.jsonl").open():
            r = json.loads(line)
            if r.get("null_alignment_type"):
                nulls += 1
                continue
            s = r.get("xcomet_xl_score")
            if s is not None:
                scores.append(float(s))
        total = len(scores) + nulls
        overall = sum(scores) / total if total else 0.0
        print("%-26s %7.4f %7.1f%% %9.4f %10d" % (
            name, overall, 100 * nulls / total, statistics.mean(scores), total))
    except Exception as exc:
        print("%-26s  ERROR %s" % (name, exc))
