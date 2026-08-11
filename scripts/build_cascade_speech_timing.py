#!/usr/bin/env python3
"""Build target-speech playout timing for a cascade run (LongYAAL / ending offset).

台账 4.-14。级联 run 的 instances.log 此前只有线性占位 delays，从未报过
延迟。本脚本复用 canonical v2 协议的全部实现
（build_acl6060_target_speech_instances：whisper-1 词时间戳、canonical
hypothesis 单调对齐、PCM FIFO playout、unit 映射），只替换"packet 到达
时刻"的来源：

- 基线（KIT/OpenAI/Gemini）：live session 实测的逐 packet arrival；
- 级联：**turn 块仿真**——turn k 的音频块在 InfiniSST 发射该 turn 文本的
  时刻整块"到达"（swrow 的 delay_ms，即源音频 chunk 边界），时长取该 run
  的实际生成音频。playout 同为 zero-jitter FIFO。

因此级联数字是 **computation-simulated 下界**：不含 InfiniSST 与 TTS 的
计算耗时、也不含网络。与基线实测数字并排时必须标注这一点。
source 侧时间线为 1.92s chunk 实时推送（源音频 chunk 边界即发送时刻）。

usage:
  build_cascade_speech_timing.py --bench <acl_bench> --tag v6 \
      --modearg slidingsoft3_speed125 --api-key-file /data/openai_key.txt
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_acl6060_target_speech_instances import (  # noqa: E402
    TIMING_METHOD,
    align_hypothesis_units,
    packet_playout_timeline,
    read_jsonl,
    read_secret,
    timed_alignment_units,
    transcribe_windows,
    unit_playout_times,
    write_json,
    write_jsonl,
)

TALK_ORDER = [268, 367, 590, 110, 117]
CHUNK_MS = 1920.0
CASCADE_SOURCE_TIMING = "realtime_192s_chunk_boundary_simulated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--modearg", required=True,
                        help="e.g. sliding_chunk192 / slidingsoft3_speed125")
    parser.add_argument("--api-key-file", required=True, type=Path)
    parser.add_argument("--asr-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--asr-model", default="gpt-4o-mini-transcribe")
    parser.add_argument("--timestamp-model", default="whisper-1")
    parser.add_argument("--window-s", type=float, default=120.0)
    parser.add_argument("--min-alignment-coverage", type=float, default=0.6)
    return parser.parse_args()


def wav_duration_ms(path: Path) -> float:
    import soundfile

    info = soundfile.info(str(path))
    return info.frames / info.samplerate * 1000.0


def main() -> None:
    args = parse_args()
    bench = args.bench.resolve()
    prefix = "chunk192"
    speed_factor = 1.0
    for sp, f in (("speed125", 1.25), ("speed150", 1.5)):
        if args.modearg.endswith("_" + sp):
            prefix, speed_factor = sp, f
    quality_rd = bench / "rundirs" / (
        f"acl6060_live_enzh_cascade_moss{args.tag}_{args.modearg}_chunk192_speed1_gptasr"
    )
    lat_rd = Path(str(quality_rd) + "_latency")
    lat_rd.mkdir(parents=True, exist_ok=True)
    # note (luojiaxuan): 延迟走独立 run dir——本脚本会用带时间戳的新 ASR 覆盖
    # instances.log，不能污染 canonical 质量 run。
    if not (lat_rd / "run_config.json").exists():
        shutil.copy2(quality_rd / "run_config.json", lat_rd / "run_config.json")
    quality_rows = {int(r["index"]): r for r in read_jsonl(quality_rd / "instances.log")}

    api_key = read_secret(args.api_key_file)
    wav_root = bench / f"tts_wavs_{args.tag}_{args.modearg}"
    out_rows, timing_rows = [], []
    for index, talk in enumerate(TALK_ORDER):
        seg_rows = read_jsonl(bench / "tts_rows" / f"talk{talk}.{prefix}.swrow.jsonl")
        segments = seg_rows[0]["segments"]
        summary = read_jsonl(wav_root / f"talk{talk}.{prefix}.summary.jsonl")[0]
        turns = summary["turns"]
        if len(turns) != len(segments):
            raise ValueError(
                f"talk{talk}: {len(turns)} turns vs {len(segments)} swrow segments")

        packets, audio_start = [], 0.0
        for i, (seg, turn) in enumerate(zip(segments, turns)):
            dur_ms = float(turn["duration_s"]) * 1000.0
            packets.append({
                "packet_index": i,
                "received_at_ms": float(seg["delay_ms"]),
                "duration_ms": round(dur_ms, 3),
                "audio_start_ms": round(audio_start, 3),
                "audio_end_ms": round(audio_start + dur_ms, 3),
            })
            audio_start += dur_ms
        packets = packet_playout_timeline(packets)

        src = quality_rows[index]["source"]
        src_path = Path(src[0] if isinstance(src, list) else src)
        # note (luojiaxuan): 速度档实际推流的是加速后的音频，swrow 的 delay_ms
        # 也在加速时钟里（最后一个 turn ≈ 原时长/速度因子，已逐 talk 核验）。
        # 质量 instances 里的 source 路径指向 1× 原始 wav，这里除以速度因子
        # 折算；此前直接用原始时长导致 ending offset 假性为负。
        source_length_ms = wav_duration_ms(src_path) / speed_factor
        n_chunks = math.ceil(source_length_ms / CHUNK_MS)
        source_timeline = [
            (k * CHUNK_MS, min(k * CHUNK_MS, source_length_ms))
            for k in range(1, n_chunks + 1)
        ]

        target_wav = Path(summary["wav"])
        sample_dir = lat_rd / "samples" / f"talk{talk}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        windows = transcribe_windows(
            audio_path=target_wav, sample_dir=sample_dir, api_key=api_key,
            base_url=args.asr_base_url, asr_model=args.asr_model,
            timestamp_model=args.timestamp_model, target_lang="zh",
            window_s=args.window_s, resume=True,
        )
        prediction = " ".join(
            str(r.get("asr_text") or "").strip()
            for r in windows if str(r.get("asr_text") or "").strip())
        timestamp_words = [w for r in windows for w in r.get("timestamp_words") or []]
        timed_units = timed_alignment_units(timestamp_words, "zh")
        audio_unit_ends, alignment = align_hypothesis_units(prediction, "zh", timed_units)
        if alignment["alignment_coverage"] < args.min_alignment_coverage:
            raise ValueError(
                f"talk{talk}: alignment coverage {alignment['alignment_coverage']:.3f}")
        delays, elapsed = unit_playout_times(
            audio_unit_ends, packets, source_timeline, source_length_ms)

        out_rows.append({
            "index": index,
            "source": quality_rows[index]["source"],
            "prediction": prediction,
            "delays": delays,
            "elapsed": elapsed,
            "prediction_length": len(elapsed),
            "source_length": round(source_length_ms, 3),
            "timing_method": TIMING_METHOD,
        })
        timing_rows.append({
            "index": index, "talk": talk,
            "target_audio": str(target_wav),
            "target_audio_duration_ms": round(audio_start, 3),
            "packet_count": len(packets),
            "last_playout_end_ms": packets[-1]["playout_end_ms"],
            "source_length_ms": round(source_length_ms, 3),
            "timing_method": TIMING_METHOD,
            "cascade_playout_simulation": True,
            "source_timing_method": CASCADE_SOURCE_TIMING,
            **alignment,
        })
        print(f"talk{talk}: units {len(elapsed)}  coverage "
              f"{alignment['alignment_coverage']:.3f}", flush=True)

    write_jsonl(lat_rd / "instances.log", out_rows)
    write_jsonl(lat_rd / "target_speech_timing.jsonl", timing_rows)
    config = json.loads((lat_rd / "run_config.json").read_text())
    config.update({
        "latency_timing_method": TIMING_METHOD,
        "source_consumption_timing_method": CASCADE_SOURCE_TIMING,
        "cascade_playout_simulation": True,
        "target_speech_asr_model": args.asr_model,
        "target_speech_timestamp_model": args.timestamp_model,
    })
    write_json(lat_rd / "run_config.json", config)
    print(json.dumps({"lat_run_dir": str(lat_rd), "samples": len(out_rows)}))


if __name__ == "__main__":
    main()
