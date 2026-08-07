#!/usr/bin/env python3
"""Build the ACL6060 En-Zh target-speech + SEGALE segmentation browser.

# note (luojiaxuan): 同事要审计两件事：(1) under-translation 里有多少其实是
# SEGALE 对齐失败（null alignment），(2) source speedup 下 BLEU 非单调的原因。
# 本脚本把 GPT/Gemini/我们级联的整场 target speech mp3 + 每个 cell 的
# SEGALE 分段中间结果（aligned_spacy_hyp.jsonl 等）复制进 Pages 发布目录，
# 并渲染成逐句可听可查的静态页面。GPT/Gemini 的 segale_alignment 已在本分支
# artifacts/ 下；级联的 rundir 在 moss-tts-infinisst 分支，需 --cascade-root
# 指向该分支的本地 checkout；mp3 由 --audio-src 提供（64kbps mono，源 wav 的
# 规范位置见 target_speech/index.html 页脚说明）。
"""
import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

TALKS = [268, 367, 590, 110, 117]
SPEED_TOKENS = {"1": "1x", "1p25": "1.25x", "1p5": "1.5x"}

REPO = Path(__file__).resolve().parents[1]
PROJ = REPO / "projects/acl6060_s2s_metrics_seed"
OUT = PROJ / "artifacts/acl6060_segale_diagnostics/target_speech"


def cells(cascade_root: Path):
    rows = []
    for sysid, label in [("openai", "GPT Realtime"), ("gemini", "Gemini Live")]:
        for tok in SPEED_TOKENS:
            rows.append({
                "key": f"enzh_{sysid}_{tok}",
                "label": f"{label} En-Zh {SPEED_TOKENS[tok]}",
                "speed": SPEED_TOKENS[tok],
                "system": label,
                "segale": PROJ / f"artifacts/acl6060_live_enzh_{sysid}_chunk960_speed{tok}/segale_alignment",
                "asr": "gpt-4o-mini-transcribe (canonical)",
            })
    cas = {
        "1": "acl6060_live_enzh_cascade_mossv3_reset_chunk192_speed1",
        "1p25": "acl6060_live_enzh_cascade_mossv3_reset_speed125_gptasr",
        "1p5": "acl6060_live_enzh_cascade_mossv3_reset_speed150_gptasr",
    }
    for tok, run in cas.items():
        rows.append({
            "key": f"enzh_cascade_{tok}",
            "label": f"Ours (InfiniSST + MOSS v3, session reset) En-Zh {SPEED_TOKENS[tok]}",
            "speed": SPEED_TOKENS[tok],
            "system": "Ours cascade v3+reset",
            "segale": cascade_root / f"projects/infinisst_moss_tts_cascade/rundirs/{run}/segale_alignment",
            "cascade_run": cascade_root / f"projects/infinisst_moss_tts_cascade/rundirs/{run}",
            "asr": "gpt-4o-mini-transcribe (canonical)",
        })
    return rows


def full_table_metrics():
    # note (luojiaxuan): GPT/Gemini 的 BLEU/XCOMET 取自 canonical
    # acl6060_full_table.tsv，避免手抄数字出错。
    out = {}
    path = PROJ / "artifacts/acl6060_full_table.tsv"
    header = None
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if header is None:
            header = parts
            continue
        row = dict(zip(header, parts))
        if row.get("Language") != "En-Zh":
            continue
        sys_name = row["System"].lower()
        sysid = "openai" if "gpt" in sys_name else ("gemini" if "gemini" in sys_name else None)
        if sysid is None:
            continue
        out[(sysid, row["Speedup"])] = (float(row["BLEU"]), float(row["XCOMET-XL"]))
    return out


def load_cell(cell, table):
    aligned = [json.loads(l) for l in (cell["segale"] / "hyp/aligned_spacy_hyp.jsonl").open()]
    transcripts = {}
    for line in (cell["segale"] / "instances.segale.jsonl").open():
        row = json.loads(line)
        transcripts[row["index"]] = row.get("prediction", "")
    if "cascade_run" in cell:
        bleu = json.load((cell["cascade_run"] / "bleu_summary.json").open())["bleu"]
        xcomet = json.load((cell["cascade_run"] / "xcomet_summary.json").open())["xcomet_xl"]
    else:
        sysid = "openai" if "openai" in cell["key"] else "gemini"
        bleu, xcomet = table[(sysid, cell["speed"])]
    nulls = sum(1 for r in aligned if not r["tgt"].strip())
    return aligned, transcripts, bleu, xcomet, nulls


def esc(s):
    return html.escape(s or "")


STYLE = """
body{font-family:Arial,sans-serif;margin:24px;color:#20242a;background:#fafbfc}
h1{margin-bottom:6px}h2{margin-top:34px}p{max-width:1200px;line-height:1.45}
table{width:100%;border-collapse:collapse;background:#fff;font-size:13px;margin-top:8px}
th,td{border:1px solid #d8dde3;padding:6px;text-align:left;vertical-align:top}
th{background:#edf1f5;position:sticky;top:0}
tr.null td{background:#fde8e8}
td.num,th.num{text-align:right}
audio{width:640px;max-width:100%}
.note{background:#fff8db;border-left:4px solid #ca8a04;padding:10px;max-width:1200px}
details{margin:8px 0;max-width:1200px}summary{cursor:pointer;color:#075985}
pre{white-space:pre-wrap;background:#f4f6f8;padding:10px;font-size:12px}
a{color:#075985}
"""


def render_cell_page(cell, aligned, transcripts, bleu, xcomet, nulls):
    by_doc = {}
    for r in aligned:
        by_doc.setdefault(r["doc_id"], []).append(r)
    parts = [f"<style>{STYLE}</style><h1>{esc(cell['label'])}</h1>"]
    parts.append(
        f"<p>SEGALE BLEU <b>{bleu:.2f}</b> · XCOMET-XL <b>{xcomet:.3f}</b> · "
        f"aligned segments {len(aligned)} · <b>null alignments {nulls}</b> "
        f"({nulls / max(len(aligned), 1):.1%}) · ASR: {esc(cell['asr'])}</p>"
        "<p class='note'>红色行 = null alignment：该 reference 句没有任何 hypothesis 对齐上，"
        "BLEU/XCOMET 按空译文计 0 分。判断是否为 SEGALE 误伤：先展开该 talk 的完整 ASR 转写"
        "（下方 details），Ctrl-F 搜 reference 的关键词；若转写里其实有对应内容，则是对齐/切分问题，"
        "而不是真正的 under-translation。音频播放器放的是整场 target speech，可对照听。</p>"
        "<p><a href='index.html'>&larr; 返回 target speech 总览</a></p>"
    )
    for idx, talk in enumerate(TALKS):
        doc = f"2022.acl-long.{talk}.wav"
        rows = by_doc.get(doc, [])
        mp3 = f"audio/{cell['key']}__talk{talk}.mp3"
        parts.append(f"<h2>2022.acl-long.{talk}</h2>")
        parts.append(f"<audio controls preload='none' src='{mp3}'></audio>")
        transcript = transcripts.get(idx, "")
        parts.append(
            f"<details><summary>完整 ASR 转写（{len(transcript)} chars，切分前原文）</summary>"
            f"<pre>{esc(transcript)}</pre></details>"
        )
        parts.append(
            "<table><tr><th>seg</th><th>source (EN)</th><th>reference (ZH)</th>"
            "<th>aligned hypothesis (ASR of target speech)</th><th class='num'>refs merged</th></tr>"
        )
        for r in rows:
            cls = " class='null'" if not r["tgt"].strip() else ""
            merged = len(r.get("src_ref_ids", []) or [])
            tgt = esc(r["tgt"]) if r["tgt"].strip() else "<i>NULL — no hypothesis aligned</i>"
            parts.append(
                f"<tr{cls}><td class='num'>{r['seg_id']}</td><td>{esc(r['src'])}</td>"
                f"<td>{esc(r['ref'])}</td><td>{tgt}</td><td class='num'>{merged}</td></tr>"
            )
        parts.append("</table>")
    parts.append(
        "<p>Raw intermediates: "
        f"<a href='segale/{cell['key']}/aligned_spacy_hyp.jsonl'>aligned_spacy_hyp.jsonl</a> · "
        f"<a href='segale/{cell['key']}/instances.segale.jsonl'>instances.segale.jsonl</a> · "
        f"<a href='segale/{cell['key']}/alignment_summary.json'>alignment_summary.json</a></p>"
    )
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-src", required=True, type=Path)
    ap.add_argument("--cascade-root", required=True, type=Path)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audio").mkdir(exist_ok=True)
    table = full_table_metrics()

    # note (luojiaxuan): --audio-src 里 GPT/Gemini mp3 名为
    # enzh_{openai,gemini}_chunk960_speed{tok}__{idx}_2022.acl-long.{talk}.mp3，
    # 级联为 enzh_cascade_mossv3reset_speed{tok}__talk{talk}.mp3；
    # 统一改名为 <cellkey>__talk<talk>.mp3 方便页面引用。
    idx_of_talk = {268: "000", 367: "001", 590: "002", 110: "003", 117: "004"}
    summary_rows = []
    for cell in cells(args.cascade_root):
        for talk in TALKS:
            if "cascade" in cell["key"]:
                tok = cell["key"].rsplit("_", 1)[1]
                src = args.audio_src / f"enzh_cascade_mossv3reset_speed{tok}__talk{talk}.mp3"
            else:
                sysid = "openai" if "openai" in cell["key"] else "gemini"
                tok = cell["key"].rsplit("_", 1)[1]
                src = args.audio_src / f"enzh_{sysid}_chunk960_speed{tok}__{idx_of_talk[talk]}_2022.acl-long.{talk}.mp3"
            dst = OUT / "audio" / f"{cell['key']}__talk{talk}.mp3"
            shutil.copy2(src, dst)
        seg_dst = OUT / "segale" / cell["key"]
        seg_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cell["segale"] / "hyp/aligned_spacy_hyp.jsonl", seg_dst / "aligned_spacy_hyp.jsonl")
        shutil.copy2(cell["segale"] / "instances.segale.jsonl", seg_dst / "instances.segale.jsonl")
        shutil.copy2(cell["segale"] / "alignment_summary.json", seg_dst / "alignment_summary.json")

        aligned, transcripts, bleu, xcomet, nulls = load_cell(cell, table)
        (OUT / f"seg_{cell['key']}.html").write_text(
            render_cell_page(cell, aligned, transcripts, bleu, xcomet, nulls), encoding="utf-8"
        )
        summary_rows.append((cell, bleu, xcomet, len(aligned), nulls))

    idx = [f"<style>{STYLE}</style><h1>ACL6060 En-Zh target speech &amp; SEGALE segmentation browser</h1>"]
    idx.append(
        "<p>每个 cell 提供整场 target speech（64kbps mono mp3）和 SEGALE 切分/对齐的逐句中间结果，"
        "用于审计：(1) under-translation 中有多少是 SEGALE 对齐失败；(2) source speedup 下 BLEU 非单调"
        "（Ours 34.69 → 32.62 → 36.14；Gemini 单调上升 40.39 → 42.16 → 43.25；GPT 单调下降）。"
        "所有 BLEU/XCOMET 均为 SEGALE 管线 + gpt-4o-mini-transcribe canonical ASR 的可比数字。</p>"
        "<p><a href='../index.html'>&larr; 返回 SEGALE diagnostics 总览</a></p>"
    )
    idx.append(
        "<table><tr><th>Cell</th><th>Speed</th><th class='num'>BLEU</th><th class='num'>XCOMET-XL</th>"
        "<th class='num'>segments</th><th class='num'>null aligns</th><th>逐句页面</th></tr>"
    )
    for cell, bleu, xcomet, nseg, nulls in summary_rows:
        idx.append(
            f"<tr><td>{esc(cell['system'])}</td><td>{esc(cell['speed'])}</td>"
            f"<td class='num'>{bleu:.2f}</td><td class='num'>{xcomet:.3f}</td>"
            f"<td class='num'>{nseg}</td><td class='num'>{nulls}</td>"
            f"<td><a href='seg_{cell['key']}.html'>audio + segmentation</a></td></tr>"
        )
    idx.append("</table>")
    idx.append(
        "<p>源 wav（未压缩 24kHz）：GPT/Gemini 来自 speech-playout raw bundle"
        "（本地 staging，PENDING_HF_UPLOAD → gavinlaw/acl6060-s2s-speech-playout-raw）；"
        "级联来自 hyper00 acl_bench tts_wavs_v3reset / tts_wavs_speed。"
        "mp3 sha256 见 <a href='audio_manifest.sha256'>audio_manifest.sha256</a>。</p>"
    )
    (OUT / "index.html").write_text("\n".join(idx), encoding="utf-8")

    manifest = []
    for p in sorted((OUT / "audio").glob("*.mp3")):
        manifest.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  audio/{p.name}")
    (OUT / "audio_manifest.sha256").write_text("\n".join(manifest) + "\n")
    print(f"cells={len(summary_rows)} mp3={len(manifest)} out={OUT}")


if __name__ == "__main__":
    main()
