"""Concatenate + Qwen3-ASR a cascade run, parameterized by (tag, mode).

# note (luojiaxuan): 与 score_base.py 同一口径（自建 Qwen3-ASR、120s 窗、
# uniform proxy delays），只是把 wav 目录参数化，用于 v4 / v3ctl x sliding /
# reset 的四个 run。缺失或失败的 session 一律跳过不补救，保持真实表现。
usage: score_generic.py <tag> <mode>
"""
import glob
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, "/data/S2S_omni/scripts")
from score_acl_cascade import transcribe_openai_windows

tag, mode = sys.argv[1], sys.argv[2]
bench = Path("/data/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench")
snap = Path(glob.glob("/root/.cache/huggingface/hub/datasets--gavinlaw--rasst-main-result-data/snapshots/*/")[0])
inputs = snap / "main_result/inputs/acl_zh"

wav_root = bench / f"tts_wavs_{tag}_{mode}"
# note (luojiaxuan): 所有 sliding* 模式（含 sliding6/sliding3 窗口扫描）都用
# swrow（整场一行）输入；只有 reset 用 rows。之前写成精确匹配 "sliding"，
# 导致 sliding6 去读 34 行的 session 文件、row_id 全对不上而静默产出空音频。
rows_suffix = "swrow" if mode.startswith("sliding") else "rows"
run_name = f"acl6060_live_enzh_cascade_moss{tag}_{mode}_chunk192_speed1"
run_dir = bench / "rundirs" / run_name
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "run_config.json").write_text(json.dumps({
    "provider": f"cascade_infinisst_moss{tag}_{mode}",
    "target_lang": "zh", "lang_code": "zh", "speed_factor": 1.0, "chunk_ms": 1920,
    "dataset_root": str(snap),
    "source_text_file": str(inputs / "source_text.txt"),
    "ref_file": str(inputs / "ref.txt"),
    "audio_yaml": str(inputs / "audio.yaml"),
    "asr": "Qwen3-ASR-1.7B-selfhosted-120swin",
    "timing_protocol": "uniform_proxy_NOT_comparable",
    "tts": f"{tag} checkpoint (Tilde), {mode} mode, fixed zh ref",
}, indent=1))

rows_out, qa = [], []
for index, talk in enumerate([268, 367, 590, 110, 117]):
    order = [json.loads(l)["row_id"]
             for l in (bench / f"tts_rows/talk{talk}.chunk192.{rows_suffix}.jsonl").open()]
    summ = {json.loads(l)["row_id"]: json.loads(l)
            for l in (wav_root / f"talk{talk}.chunk192.summary.jsonl").open()}
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
    wav_path = wav_root / f"talk{talk}.chunk192.full.wav"
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
    prediction = transcribe_openai_windows(
        wav_path, key=None, base_url="http://127.0.0.1:47500", model="Qwen/Qwen3-ASR-1.7B")
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
