"""Concatenate + Qwen3-ASR a cascade run, parameterized by (tag, mode).

# note (luojiaxuan): 与 score_base.py 同一口径（自建 Qwen3-ASR、120s 窗、
# uniform proxy delays），只是把 wav 目录参数化，用于 v4 / v3ctl x sliding /
# reset 的四个 run。缺失或失败的 session 一律跳过不补救，保持真实表现。
usage: score_generic.py <tag> <mode>
"""
import glob
import os
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, "/data/S2S_omni/scripts")
from score_acl_cascade import transcribe_openai_windows

tag, mode = sys.argv[1], sys.argv[2]
# note (luojiaxuan): 第三个参数选 ASR 后端。qwen3 = 自建（免费，日常回归）；
# gpt = canonical gpt-4o-mini-transcribe（付费，正式对比口径）。两者不可混比。
asr_backend = sys.argv[3] if len(sys.argv) > 3 else "qwen3"
if asr_backend == "gpt":
    _key = Path("/data/openai_key.txt").read_text().strip()
    _asr_kwargs = dict(key=_key)
    _asr_name = "gpt-4o-mini-transcribe-120swin"
    _suffix = "_gptasr"
else:
    _asr_kwargs = dict(key=None, base_url="http://127.0.0.1:47500",
                       model="Qwen/Qwen3-ASR-1.7B")
    _asr_name = "Qwen3-ASR-1.7B-selfhosted-120swin"
    _suffix = ""
bench = Path("/data/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench")
snap = Path(glob.glob("/root/.cache/huggingface/hub/datasets--gavinlaw--rasst-main-result-data/snapshots/*/")[0])
inputs = snap / "main_result/inputs/acl_zh"

wav_root = bench / f"tts_wavs_{tag}_{mode}"
# note (luojiaxuan): 所有 sliding* 模式（含 sliding6/sliding3 窗口扫描）都用
# swrow（整场一行）输入；只有 reset 用 rows。之前写成精确匹配 "sliding"，
# 导致 sliding6 去读 34 行的 session 文件、row_id 全对不上而静默产出空音频。
# note (luojiaxuan): 整场单行（swrow）的模式有 sliding* 和 guard*；
# 只有 reset/anchor 用分段的 rows。漏掉 guard 会读错输入、静默产出空音频。
rows_suffix = "swrow" if mode.startswith(("sliding", "guard")) else "rows"
# note (luojiaxuan): mode 形如 "sliding" / "reset" / "sliding_speed125"。
# 后一种带速度档，输入文件与 wav 文件名前缀要跟着换成 speed125，
# 之前这里写死 chunk192，速度档一律读错文件。
# note (luojiaxuan): PREFIX 决定读哪一份 rows 与 summary。默认按 mode 后缀推
# 速度档；换了文本来源（例如重训后的 InfiniSST）时 rows 是另一套文件，用
# PREFIX 环境变量显式指定，与 run_eval_queue.sh 的同名变量保持一致。
PREFIX = "chunk192"
for _sp in ("speed125", "speed150"):
    if mode.endswith("_" + _sp):
        PREFIX = _sp
        break
PREFIX = os.environ.get("PREFIX", PREFIX)
run_name = f"acl6060_live_enzh_cascade_moss{tag}_{mode}_chunk192_speed1{_suffix}"
run_dir = bench / "rundirs" / run_name
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "run_config.json").write_text(json.dumps({
    "provider": f"cascade_infinisst_moss{tag}_{mode}",
    "target_lang": "zh", "lang_code": "zh", "speed_factor": 1.0, "chunk_ms": 1920,
    "dataset_root": str(snap),
    "source_text_file": str(inputs / "source_text.txt"),
    "ref_file": str(inputs / "ref.txt"),
    "audio_yaml": str(inputs / "audio.yaml"),
    "asr": _asr_name,
    "timing_protocol": "uniform_proxy_NOT_comparable",
    "tts": f"{tag} checkpoint (Tilde), {mode} mode, fixed zh ref",
}, indent=1))

rows_out, qa = [], []
for index, talk in enumerate([268, 367, 590, 110, 117]):
    order = [json.loads(l)["row_id"]
             for l in (bench / f"tts_rows/talk{talk}.{PREFIX}.{rows_suffix}.jsonl").open()]
    summ = {json.loads(l)["row_id"]: json.loads(l)
            for l in (wav_root / f"talk{talk}.{PREFIX}.summary.jsonl").open()}
    pcm, rate, missing = [], None, 0
    for rid in order:
        row = summ.get(rid)
        path = Path(row["wav"]) if row and row.get("wav") else None
        if path is None or not path.exists():
            missing += 1
            continue
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            pcm.append(w.readframes(w.getnframes()))
    rate = rate or 24000
    wav_path = wav_root / f"talk{talk}.{PREFIX}.full.wav"
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        for c in pcm:
            w.writeframes(c)
    target_s = sum(len(c) for c in pcm) / 2 / rate
    source_wav = snap / f"main_result/audio/acl6060/2022.acl-long.{talk}.wav"
    with wave.open(str(source_wav), "rb") as w:
        source_ms = w.getnframes() / w.getframerate() * 1000.0
    prediction = transcribe_openai_windows(wav_path, **_asr_kwargs)
    units = [ch for ch in prediction if not ch.isspace()]
    n = max(1, len(units))
    delays = [round((i + 1) / n * source_ms, 3) for i in range(len(units))]
    rows_out.append({"index": index, "source": [str(source_wav)], "prediction": prediction,
                     "delays": delays, "elapsed": delays, "durations": [],
                     "target_duration_s": round(target_s, 3)})
    qa.append({"talk": talk, "rows": len(order), "missing_rows": missing,
               "chars": len(units), "target_s": round(target_s, 1)})
    print(json.dumps(qa[-1]), flush=True)

with (run_dir / "instances.log").open("w", encoding="utf-8") as out:
    for row in rows_out:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
(run_dir / "session_qa.json").write_text(json.dumps({"per_talk": qa}, indent=1))
print("RUNDIR_DONE", run_name)
