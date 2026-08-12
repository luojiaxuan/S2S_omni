#!/usr/bin/env python3
"""Re-score a baseline system (KIT/OpenAI/Gemini) under the new canonical op.

台账 4.-17。新口径（用户裁定）：Qwen3-ASR（自托管 47500）+ SEGALE 句级
BLEU（null 保留为空假设）+ XCOMET-XL reference-based（null 置零）。
本脚本只做第 1 步——用 Qwen3-ASR 重转写基线的 target wav 并构造 run dir；
其后的 SEGALE/BLEU/XCOMET 步骤复用 score_chain_refbased.sh 的 2–5 步。

wav 来源：gavinlaw/acl6060-s2s-speech-playout-raw（Mac 暂存清欠账后的
canonical 位置），本地放 /data/playout_raw/。
instances 只需 index/source(snap 路径)/prediction + 线性占位 delays——
质量链不读时序；延迟侧基线沿用既有 live 实测数字，不在本口径重算。

usage: score_baseline_newop.py <kit|openai|gemini> <speed1|speed1p25|speed1p5>
"""
from __future__ import annotations

import glob
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, "/data/S2S_omni/scripts")
from score_acl_cascade import transcribe_openai_windows  # noqa: E402

TALK_ORDER = [268, 367, 590, 110, 117]
RAW = Path("/data/playout_raw")
BENCH = Path("/data/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench")


def main() -> None:
    system, speed = sys.argv[1], sys.argv[2]
    sub = "kit" if system == "kit" else "openai_gemini"
    cell = RAW / sub / f"enzh_{system}_chunk960_{speed}"
    snap = Path(glob.glob(
        "/root/.cache/huggingface/hub/datasets--gavinlaw--rasst-main-result-data/snapshots/*/")[0])
    inputs = snap / "main_result/inputs/acl_zh"

    run_name = f"acl6060_live_enzh_{system}_chunk960_{speed}_qwen3newop"
    run_dir = BENCH / "rundirs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir_cfg = {
        "provider": f"baseline_{system}",
        "target_lang": "zh", "lang_code": "zh",
        "speed_factor": 1.0, "chunk_ms": 960,
        "dataset_root": str(snap),
        "source_text_file": str(inputs / "source_text.txt"),
        "ref_file": str(inputs / "ref.txt"),
        "audio_yaml": str(inputs / "audio.yaml"),
        "asr": "Qwen3-ASR-1.7B-selfhosted-120swin",
        "candidate_text_source": "target_speech_asr_qwen3",
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_dir_cfg, indent=1))

    rows_out = []
    for index, talk in enumerate(TALK_ORDER):
        sample = cell / f"{index:03d}_2022.acl-long.{talk}"
        wav = sample / "target_tts.wav"
        if not wav.exists():
            cands = sorted(sample.glob("target_audio_*.wav"))
            if len(cands) != 1:
                raise FileNotFoundError(f"{sample}: expected 1 target wav, got {cands}")
            wav = cands[0]
        prediction = transcribe_openai_windows(
            wav, key=None, base_url="http://127.0.0.1:47500", model="Qwen/Qwen3-ASR-1.7B")
        source_wav = snap / f"main_result/audio/acl6060/2022.acl-long.{talk}.wav"
        with wave.open(str(source_wav), "rb") as w:
            source_ms = w.getnframes() / w.getframerate() * 1000.0
        units = [ch for ch in prediction if not ch.isspace()]
        n = max(1, len(units))
        rows_out.append({
            "index": index,
            "source": [str(source_wav)],
            "prediction": prediction,
            "delays": [round((i + 1) / n * source_ms, 3) for i in range(len(units))],
            "elapsed": [round((i + 1) / n * source_ms, 3) for i in range(len(units))],
            "durations": [],
        })
        print(f"talk{talk}: {len(units)} chars", flush=True)

    with (run_dir / "instances.log").open("w", encoding="utf-8") as out:
        for row in rows_out:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("BASELINE_ASR_DONE", run_name)


if __name__ == "__main__":
    main()
