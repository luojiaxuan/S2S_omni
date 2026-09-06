"""Aggregate the protocol-A metrics.json files into a table: deterministic A/B per thinker, and the
sampled runs as mean and spread. Run inside the aries container; reads /data/serving_ab/results."""
import glob
import json
import os
import statistics as st

JOBS = {
    "theirs": {"det": 92001, "s": [92002, 92003, 92004]},
    "word": {"det": 92011, "s": [92012, 92013, 92014]},
    "phrase": {"det": 92021, "s": [92022, 92023, 92024]},
}
NAMES = {"theirs": "theirs phrase (owaski)", "word": "ours word-aligned", "phrase": "ours phrase-gated"}
ROOT = "/data/serving_ab/results"


def load(job):
    p = os.path.join(ROOT, f"s2st_moss-delta_dev_1920ms_{job}", "metrics.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["metrics"]


def cell(vals):
    if not vals:
        return "  -   "
    if len(vals) == 1:
        return f"{vals[0]:6.2f}"
    return f"{st.mean(vals):6.2f}±{(max(vals) - min(vals)) / 2:4.2f}"


print(f"{'thinker':22s} {'setting':12s} {'reg':3s} {'BLEU':>13s} {'XCOMET':>13s} {'EndOff ms':>13s} {'empty':>6s}")
for key in ("theirs", "word", "phrase"):
    for setting, jobs in (("deterministic", [JOBS[key]["det"]]), ("sampled x%d" % len(JOBS[key]["s"]), JOBS[key]["s"])):
        ms = [load(j) for j in jobs]
        ms = [m for m in ms if m]
        if not ms:
            print(f"{NAMES[key]:22s} {setting:12s}  (no runs yet)")
            continue
        for reg in ("CU", "CA"):
            bleu = [m[reg]["BLEU"] for m in ms]
            xc = [m[reg]["XCOMET_XL"] for m in ms]
            eo = [m[reg]["Ending_Offset"] for m in ms]
            emp = [m[reg]["n_empty_pred"] for m in ms]
            print(f"{NAMES[key]:22s} {setting:12s} {reg:3s} {cell(bleu):>13s} {cell(xc):>13s} {cell(eo):>13s} {int(st.mean(emp)):>6d}")
