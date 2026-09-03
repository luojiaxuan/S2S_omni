#!/usr/bin/env python3
"""Per-turn Qwen3-ASR of an external SimulEval speech-to-speech run, in the same shape
w1_ap.py produces for the cascade arms, so w3/w4/w5 score it under the identical
protocol (turn-wise ASR joined with "。", SEGALE alignment, SacreBLEU zh).

A turn is one SimulEval write: the rendered wavs/<i>_pred.wav is sliced at the
`intervals` starts recorded in instances.log (each turn keeps its trailing silence up
to the next piece's start), mirroring the cascade's one-turn-per-write ASR granularity.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
import wave
from pathlib import Path

TALKS = (268, 110, 117)


def load_helper(path: Path):
    spec = importlib.util.spec_from_file_location("acl_cascade_transcribe_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_turn_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ext-dir", required=True, type=Path,
                        help="directory holding instances.log and wavs/<i>_pred.wav")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--transcribe-helper", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:47500")
    parser.add_argument("--label", required=True)
    parser.add_argument("--merge-window-s", type=float, default=0.0,
                        help="merge consecutive pieces into ASR windows spanning at least "
                             "this many seconds (0 = one ASR call per piece)")
    args = parser.parse_args()

    helper = load_helper(args.transcribe_helper)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    inputs = args.dataset_root / "main_result/inputs/acl_zh"
    config = {
        "provider": f"external_{args.label}",
        "target_lang": "zh",
        "lang_code": "zh",
        "speed_factor": 1.0,
        "chunk_ms": 2000,
        "asr_merge_window_s": args.merge_window_s,
        "dataset_root": str(args.dataset_root),
        "source_text_file": str(inputs / "source_text.txt"),
        "ref_file": str(inputs / "ref.txt"),
        "audio_yaml": str(inputs / "audio.yaml"),
        "asr": "Qwen3-ASR-1.7B-selfhosted+perturn-forced-boundary",
        "timing_protocol": "uniform_proxy_NOT_comparable",
        "tts": f"{args.label}: SimulEval speech output, one turn per write",
    }
    (args.run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    ext_rows = {}
    for line in (args.ext_dir / "instances.log").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            ext_rows[Path(row["source"]).stem] = row

    cache_root = args.run_dir / "asr_cache"
    cache_root.mkdir(exist_ok=True)
    tmp_root = args.run_dir / "tmp"
    tmp_root.mkdir(exist_ok=True)
    rows_out, qa = [], []
    for index, talk in enumerate(TALKS):
        row = ext_rows[f"2022.acl-long.{talk}"]
        with wave.open(str(args.ext_dir / "wavs" / f"{row['index']}_pred.wav"), "rb") as handle:
            sample_rate = handle.getframerate()
            pcm = handle.readframes(handle.getnframes())
        total = len(pcm) // 2
        starts = [int(round(start_ms / 1000.0 * sample_rate)) for start_ms, _ in row["intervals"]]
        if args.merge_window_s > 0:
            # Windows of consecutive pieces spanning >= merge_window_s, cut only at piece starts.
            min_len = int(args.merge_window_s * sample_rate)
            merged = [starts[0]]
            for s in starts[1:]:
                if s - merged[-1] >= min_len:
                    merged.append(s)
            starts = merged
        bounds = starts + [total]
        texts = []
        with (cache_root / f"talk{talk}.jsonl").open("w", encoding="utf-8") as cache_handle:
            for turn_index in range(len(starts)):
                a, b = bounds[turn_index], bounds[turn_index + 1]
                turn_pcm = pcm[2 * a:2 * b]
                if b <= a:
                    text = ""
                else:
                    with tempfile.NamedTemporaryFile(dir=tmp_root, suffix=".wav", delete=False) as tmp:
                        tmp_path = Path(tmp.name)
                    try:
                        write_turn_wav(tmp_path, turn_pcm, sample_rate)
                        text = helper.transcribe_openai_windows(
                            tmp_path, key=None, base_url=args.base_url,
                            model="Qwen/Qwen3-ASR-1.7B").strip()
                    finally:
                        tmp_path.unlink(missing_ok=True)
                cache_handle.write(json.dumps({
                    "turn_index": turn_index, "samples": b - a,
                    "pcm_sha256": hashlib.sha256(turn_pcm).hexdigest(), "text": text,
                }, ensure_ascii=False) + "\n")
                texts.append(text)
        prediction = "".join(text + "。" for text in texts if text)
        source_wav = args.dataset_root / f"main_result/audio/acl6060/2022.acl-long.{talk}.wav"
        with wave.open(str(source_wav), "rb") as handle:
            source_ms = handle.getnframes() / handle.getframerate() * 1000.0
        units = [char for char in prediction if not char.isspace()]
        count = max(1, len(units))
        delays = [round((unit + 1) / count * source_ms, 3) for unit in range(len(units))]
        rows_out.append({"index": index, "source": [str(source_wav)], "prediction": prediction,
                         "delays": delays, "elapsed": delays})
        qa.append({"talk": talk, "turns": len(starts), "chars": len(units),
                   "target_s": round(total / sample_rate, 3),
                   "source_s": round(source_ms / 1000.0, 3)})
        print(json.dumps(qa[-1]), flush=True)

    with (args.run_dir / "instances.log").open("w", encoding="utf-8") as handle:
        for out in rows_out:
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
    (args.run_dir / "session_qa.json").write_text(json.dumps({"per_talk": qa}, indent=2) + "\n")
    print(f"EXT_RUNDIR_DONE {args.run_dir}", flush=True)


if __name__ == "__main__":
    main()
