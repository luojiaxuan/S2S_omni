#!/usr/bin/env python3
"""Document-level concatenated BLEU: 整场假设串 vs 整场参考串，不依赖任何对齐。

台账 4.-15。动机：SEGALE 口径下"漏译句保留为空假设"与 Open-LiveTranslate
上游"漏译句剔除"两种处理差 1.5–3 BLEU，争论的仲裁口径就是 BLEU 的原始
定义——全文档拼接直接算。它完全绕开切分与对齐，漏译通过长度比/brevity
penalty 自然计入，任何一方都无从质疑。

输入两种布局：
- 级联 run：instances.log 每 talk 一行（prediction = 整场 ASR），参考句
  从任一 run 的 SEGALE 对齐产物取（金标对所有 run 相同），行序即
  TALK_ORDER；
- 基线 run（metrics-seed artifacts）：instances.log 每行自带整场
  prediction 与 reference。

usage:
  compute_doc_bleu.py --refs refs.jsonl --cascade name=path.instances.log ... \
      --baseline name=artifact_dir ...
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from sacrebleu import corpus_bleu

TALK_ORDER = [268, 367, 590, 110, 117]


def load_refs(path: Path) -> dict[str, str]:
    per_doc: dict[str, dict[int, str]] = collections.defaultdict(dict)
    for line in path.open():
        row = json.loads(line)
        per_doc[row["doc_id"]][int(row["seg_id"])] = str(row["ref"])
    return {
        doc: "".join(text for _sid, text in sorted(segs.items()))
        for doc, segs in per_doc.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs", type=Path, required=True,
                        help="aligned_spacy_hyp.jsonl carrying (doc_id, seg_id, ref)")
    parser.add_argument("--cascade", nargs="*", default=[],
                        help="name=instances.log (5 rows, TALK_ORDER)")
    parser.add_argument("--baseline", nargs="*", default=[],
                        help="name=artifact_dir (instances.log rows carry reference)")
    parser.add_argument("--tokenize", default="zh")
    args = parser.parse_args()

    doc_refs = load_refs(args.refs)
    results = []

    for spec in args.cascade:
        name, path = spec.split("=", 1)
        rows = [json.loads(l) for l in open(path)]
        docs = [f"2022.acl-long.{n}.wav" for n in TALK_ORDER]
        hyps = ["".join(str(r.get("prediction") or "").split()) for r in rows]
        refs = [doc_refs[d] for d in docs[: len(rows)]]
        score = corpus_bleu(hyps, [refs], tokenize=args.tokenize).score
        results.append((name, score, len(rows)))

    for spec in args.baseline:
        name, dirpath = spec.split("=", 1)
        rows = [json.loads(l) for l in (Path(dirpath) / "instances.log").open()]
        hyps = ["".join(str(r.get("prediction") or "").split()) for r in rows]
        refs = [str(r.get("reference") or "") for r in rows]
        score = corpus_bleu(hyps, [refs], tokenize=args.tokenize).score
        results.append((name, score, len(rows)))

    for name, score, n in results:
        print(f"DOC_BLEU {name:32s} {score:6.2f}  (docs {n})")


if __name__ == "__main__":
    main()
