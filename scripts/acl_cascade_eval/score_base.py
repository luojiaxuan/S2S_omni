"""Concatenate + score the un-finetuned MOSS-TTS-Realtime cascade baseline.

# note (luojiaxuan): 结构照抄 score_v3reset.py，两处不同：读 tts_wavs_base，
# ASR 换成自建 Qwen3-ASR（免费，与 v3 的 qwen3asr run 同口径，可直接对照
# v3 的 BLEU 29.75）。失败/runaway 的 session 没有 wav，按缺失跳过——这正是
# baseline 的真实表现，不做任何补救，否则会掩盖未微调模型的崩溃率。
"""
import glob
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, "/data/S2S_omni/scripts")
from score_acl_cascade import transcribe_openai_windows

bench = Path("/data/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench")
snap = Path(glob.glob("/root/.cache/huggingface/hub/datasets--gavinlaw--rasst-main-result-data/snapshots/*/")[0])
inputs = snap / "main_result/inputs/acl_zh"

run_name = "acl6060_live_enzh_cascade_mossbase_reset_chunk192_speed1"
run_dir = bench / "rundirs" / run_name
run_dir.mkdir(parents=True, exist_ok=True)
config = {
    "provider": "cascade_infinisst_mossbase_sessionreset",
    "target_lang": "zh", "lang_code": "zh", "speed_factor": 1.0, "chunk_ms": 1920,
    "dataset_root": str(snap),
    "source_text_file": str(inputs / "source_text.txt"),
    "ref_file": str(inputs / "ref.txt"),
    "audio_yaml": str(inputs / "audio.yaml"),
    "asr": "Qwen3-ASR-1.7B-selfhosted-120swin",
    "timing_protocol": "uniform_proxy_NOT_comparable",
    "tts": "UNFINETUNED OpenMOSS-Team/MOSS-TTS-Realtime, session reset 11 turns, fixed zh ref",
}
(run_dir / "run_config.json").write_text(json.dumps(config, indent=1))

rows_out = []
qa = []
for index, talk in enumerate([268, 367, 590, 110, 117]):
    order = [json.loads(l)["row_id"] for l in (bench / f"tts_rows/talk{talk}.chunk192.rows.jsonl").open()]
    summ = {json.loads(l)["row_id"]: json.loads(l)
            for l in (bench / f"tts_wavs_base/talk{talk}.chunk192.summary.jsonl").open()}
    pcm, rate = [], None
    missing = 0
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
    wav_path = bench / f"tts_wavs_base/talk{talk}.chunk192.full.wav"
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
    qa.append({"talk": talk, "sessions": len(order), "missing_sessions": missing,
               "chars": len(units), "target_s": round(target_s, 1)})
    print(json.dumps(qa[-1]), flush=True)

with (run_dir / "instances.log").open("w", encoding="utf-8") as out:
    for row in rows_out:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
(run_dir / "session_qa.json").write_text(json.dumps({
    "per_talk": qa,
    "total_sessions": sum(q["sessions"] for q in qa),
    "total_missing_sessions": sum(q["missing_sessions"] for q in qa),
}, indent=1))
print("BASE_RUNDIR_DONE", run_name)
