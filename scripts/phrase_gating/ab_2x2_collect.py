"""Collect the 2x2 interaction test the external review asked for (docs/reviews/20260912-*).

The 2026-09-06 deterministic runs were trained under ms-swift's default loss, where an
intermediate EMPTY assistant turn carries no supervised token. Those runs are not discarded:
they are the old-loss cells of a 2x2, and the quantity that actually supports the attribution
is not P_fixed > W_fixed but the interaction

    delta_word   = M(W_fixed) - M(W_old)
    delta_phrase = M(P_fixed) - M(P_old)
    I            = delta_phrase - delta_word

A materially positive I says the loss defect damaged the phrase arm disproportionately, which
is what the 2.69x empty-turn exposure (29.3% vs 10.9%) predicts.

Each cell is a run directory holding metrics.json and render_report.json. CU only: CA is
computation-aware and therefore not comparable across the A6000 and H200 hosts.
"""
import argparse
import json
import os

CELLS = ("W_old", "P_old", "W_fixed", "P_fixed")


def load(run_dir):
    with open(os.path.join(run_dir, "metrics.json")) as f:
        cu = json.load(f)["metrics"]["CU"]
    docs = {}
    report = os.path.join(run_dir, "render_report.json")
    if os.path.exists(report):
        with open(report) as f:
            docs = json.load(f).get("documents", {})
    return cu, docs


def empty_share(doc, chunk_s):
    """Share of policy calls that produced no audio: the silence-behaviour diagnostic."""
    total = doc["source_length_ms"] / (chunk_s * 1000.0)
    spoke = doc.get("chunks")
    if not spoke or total <= 0:
        return None
    return max(0.0, 1.0 - spoke / total)


ap = argparse.ArgumentParser()
for c in CELLS:
    ap.add_argument("--" + c.lower(), required=True, help=f"run dir for {c}")
ap.add_argument("--chunk-s", type=float, default=1.92)
ap.add_argument("--out", default="")
a = ap.parse_args()

cells = {c: load(getattr(a, c.lower())) for c in CELLS}
print(f"{'cell':10s} {'BLEU':>7s} {'XCOMET':>8s} {'EndOff ms':>10s} {'empty calls':>12s}")
summary = {}
for c in CELLS:
    cu, docs = cells[c]
    shares = [s for s in (empty_share(d, a.chunk_s) for d in docs.values()) if s is not None]
    mean_share = sum(shares) / len(shares) if shares else float("nan")
    summary[c] = {"BLEU": cu["BLEU"], "XCOMET": cu["XCOMET_XL"],
                  "Ending_Offset": cu["Ending_Offset"], "empty_call_share": mean_share,
                  "per_talk": {k: v.get("chunks") for k, v in docs.items()}}
    print(f"{c:10s} {cu['BLEU']:7.2f} {cu['XCOMET_XL']:8.4f} {cu['Ending_Offset']:10.0f} {mean_share:11.1%}")

for metric, key in (("BLEU", "BLEU"), ("XCOMET", "XCOMET")):
    dw = summary["W_fixed"][key] - summary["W_old"][key]
    dp = summary["P_fixed"][key] - summary["P_old"][key]
    print(f"\n{metric}: delta_word {dw:+.3f} | delta_phrase {dp:+.3f} | I = {dp - dw:+.3f}")
    summary.setdefault("interaction", {})[metric] = {"delta_word": dw, "delta_phrase": dp, "I": dp - dw}

pf, wf = summary["P_fixed"], summary["W_fixed"]
print(f"\nP_fixed - W_fixed: BLEU {pf['BLEU'] - wf['BLEU']:+.2f}, XCOMET {pf['XCOMET'] - wf['XCOMET']:+.4f}")
print("predeclared reading: I>0 AND P_fixed>W_fixed on both metrics -> phrase gating helps under corrected "
      "empty-turn supervision; metrics split -> report a trade-off only; both still worse -> the bug is not a "
      "sufficient explanation, investigate the gating rule next.")
print("seed rule: |P_fixed - W_fixed| < 1 BLEU or contradictory metrics -> >=3 matched training seeds before claiming.")

if a.out:
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print("wrote", a.out)
