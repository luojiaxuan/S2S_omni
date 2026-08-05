#!/usr/bin/env python3
"""Package cascade TTS outputs as ACL6060 run dirs for the SEGALE pipeline.

Per (talk, chunk) run: concatenate session wavs to one talk wav, transcribe
with gpt-4o-mini-transcribe (<=120s windows, canonical ASR), and emit a run
dir with run_config.json + instances.log in the layout expected by
build_acl6060_segale_inputs.py.

# note (luojiaxuan): `delays`/`elapsed` are uniform proxies over the source
# duration so the builder's per-unit length validation passes; quality columns
# (SEGALE BLEU / XCOMET) are unaffected, latency columns from these runs are
# NOT comparable to the canonical speech-playout protocol.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from score_acl_cascade import concat_run, transcribe_openai_windows  # noqa: E402

DOC_ORDER = [268, 367, 590, 110, 117]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", required=True)
    parser.add_argument("--dataset-root", required=True, help="HF rasst-main-result-data snapshot root")
    parser.add_argument("--openai-key-file", required=True)
    parser.add_argument("--output-base", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bench = Path(args.bench_dir)
    dataset_root = Path(args.dataset_root)
    key = Path(args.openai_key_file).read_text().strip()
    inputs = dataset_root / "main_result/inputs/acl_zh"

    for chunk in ("096", "192"):
        run_dir = Path(args.output_base) / f"acl6060_live_enzh_cascade_mossv2_chunk{chunk}_speed1"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "provider": "cascade_infinisst_mossv2",
            "target_lang": "zh",
            "lang_code": "zh",
            "speed_factor": 1.0,
            "chunk_ms": 960 if chunk == "096" else 1920,
            "dataset_root": str(dataset_root),
            "source_text_file": str(inputs / "source_text.txt"),
            "ref_file": str(inputs / "ref.txt"),
            "audio_yaml": str(inputs / "audio.yaml"),
            "asr": "gpt-4o-mini-transcribe-120swin",
            "timing_protocol": "uniform_proxy_NOT_comparable",
            "tts": "moss-tts-realtime-infinisst-en-zh-v2-multiturn, fixed zh ref, session reset 11 turns",
        }
        (run_dir / "run_config.json").write_text(json.dumps(config, indent=1))
        rows = []
        for index, talk in enumerate(DOC_ORDER):
            wav_path, target_s = concat_run(bench, talk, chunk)
            source_wav = dataset_root / f"main_result/audio/acl6060/2022.acl-long.{talk}.wav"
            with wave.open(str(source_wav), "rb") as handle:
                source_ms = handle.getnframes() / handle.getframerate() * 1000.0
            prediction = transcribe_openai_windows(wav_path, key)
            units = [ch for ch in prediction if not ch.isspace()]
            n = max(1, len(units))
            delays = [round((i + 1) / n * source_ms, 3) for i in range(len(units))]
            rows.append(
                {
                    "index": index,
                    "source": [str(source_wav)],
                    "prediction": prediction,
                    "delays": delays,
                    "elapsed": delays,
                    "durations": [],
                    "target_duration_s": round(target_s, 3),
                }
            )
            print(
                json.dumps(
                    {"run": run_dir.name, "talk": talk, "chars": len(units), "target_s": round(target_s, 2)}
                ),
                flush=True,
            )
        with (run_dir / "instances.log").open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("RUNDIRS_DONE")


if __name__ == "__main__":
    main()
