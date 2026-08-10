"""在同一批源句上配对比较两个 run，去掉漏译率带来的选择效应。

# note (luojiaxuan): 直接比「非空段均分」是有偏的——一个 run 如果把难句
# 都判成漏译，剩下的均分反而更高。这里按 (doc_id, 源句 id) 把两个 run 的
# 分数对齐，只在**两边都有分**的源句上比，并单列各自独有的漏译。
"""
import json
import statistics
import sys
from pathlib import Path

BENCH = Path("/data/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench/rundirs")


def load(run):
    """源句 id -> 分数；漏译记 0，与打分口径一致。"""
    out = {}
    for line in (BENCH / run / "xcomet_segments.jsonl").open():
        r = json.loads(line)
        s = 0.0 if r.get("null_alignment_type") else r.get("xcomet_xl_score")
        if s is None:
            continue
        for sid in (r.get("source_segment_ids") or []):
            out[(r["doc_id"], sid)] = max(out.get((r["doc_id"], sid), 0.0), float(s))
    return out


(na, ra), (nb, rb) = (a.split("=", 1) for a in sys.argv[1:3])
A, B = load(ra), load(rb)
both = sorted(set(A) & set(B))
print(f"{na}  vs  {nb}")
print(f"  两边都覆盖到的源句: {len(both)}  (A 独有 {len(set(A)-set(B))}, B 独有 {len(set(B)-set(A))})")
da = [A[k] for k in both]
db = [B[k] for k in both]
diff = [y - x for x, y in zip(da, db)]
se = statistics.stdev(diff) / len(diff) ** 0.5
print(f"  配对均分   A {statistics.mean(da):.4f}   B {statistics.mean(db):.4f}")
print(f"  配对差值   {statistics.mean(diff):+.4f} ± {se:.4f}  (t={statistics.mean(diff)/se:+.2f}, n={len(diff)})")
win = sum(1 for d in diff if d > 0.05)
lose = sum(1 for d in diff if d < -0.05)
print(f"  逐句胜负   B 更好 {win}   A 更好 {lose}   基本持平 {len(diff)-win-lose}")
