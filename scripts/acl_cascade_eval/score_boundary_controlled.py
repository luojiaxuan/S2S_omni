#!/usr/bin/env python3
"""Transcribe exact TTS turn spans and build a boundary-controlled ACL6060 run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
import wave
from pathlib import Path
from types import ModuleType

TALKS = (268, 367, 590, 110, 117)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-dir", required=True, type=Path)
    parser.add_argument("--wav-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--transcribe-helper", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:47500")
    parser.add_argument("--label", required=True)
    return parser.parse_args()


def load_helper(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("acl_cascade_transcribe_helper", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_turn_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def load_cache(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {int(record["turn_index"]): record for record in records}


def main() -> None:
    args = parse_args()
    helper = load_helper(args.transcribe_helper)
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    snap = args.dataset_root
    inputs = snap / "main_result/inputs/acl_zh"
    config = {
        "provider": f"cascade_infinisst_moss_{args.label}_boundary_controlled",
        "target_lang": "zh",
        "lang_code": "zh",
        "speed_factor": 1.0,
        "chunk_ms": 1920,
        "dataset_root": str(snap),
        "source_text_file": str(inputs / "source_text.txt"),
        "ref_file": str(inputs / "ref.txt"),
        "audio_yaml": str(inputs / "audio.yaml"),
        "asr": "Qwen3-ASR-1.7B-selfhosted+perturn-forced-boundary",
        "timing_protocol": "uniform_proxy_NOT_comparable",
        "tts": f"{args.label}, continuous codec context",
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    rows_out: list[dict] = []
    qa: list[dict] = []
    cache_root = args.wav_dir / "bc_asr_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    tmp_root = args.wav_dir / "bc_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    for index, talk in enumerate(TALKS):
        rows_path = args.rows_dir / f"talk{talk}.jsonl"
        input_rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        if len(input_rows) != 1:
            raise RuntimeError(f"talk{talk}: expected one swrow, got {len(input_rows)}")
        summaries = [
            json.loads(line)
            for line in (args.wav_dir / f"talk{talk}.summary.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if len(summaries) != 1 or summaries[0].get("failure") is not None:
            raise RuntimeError(f"talk{talk}: incomplete synthesis summary")
        summary = summaries[0]
        if summary.get("codec_context") != "continuous_per_row":
            raise RuntimeError(f"talk{talk}: synthesis did not preserve codec context")
        turns = summary["turns"]
        if len(turns) != len(input_rows[0]["segments"]):
            raise RuntimeError(f"talk{talk}: turn count mismatch")

        wav_path = Path(summary["wav"])
        with wave.open(str(wav_path), "rb") as handle:
            sample_rate = handle.getframerate()
            pcm = handle.readframes(handle.getnframes())
        turn_samples = [int(turn["samples"]) for turn in turns]
        if sum(turn_samples) * 2 != len(pcm):
            raise RuntimeError(f"talk{talk}: turn samples do not cover the full wav")

        cache_path = cache_root / f"talk{talk}.jsonl"
        cache = load_cache(cache_path)
        texts: list[str] = []
        cache_hits = 0
        offset = 0
        with cache_path.open("a", encoding="utf-8") as cache_handle:
            for turn_index, samples in enumerate(turn_samples):
                turn_pcm = pcm[offset * 2 : (offset + samples) * 2]
                offset += samples
                digest = hashlib.sha256(turn_pcm).hexdigest()
                cached = cache.get(turn_index)
                if cached is not None and cached["pcm_sha256"] == digest:
                    text = cached["text"]
                    cache_hits += 1
                elif samples == 0:
                    text = ""
                else:
                    with tempfile.NamedTemporaryFile(
                        dir=tmp_root, suffix=".wav", delete=False
                    ) as tmp:
                        tmp_path = Path(tmp.name)
                    try:
                        write_turn_wav(tmp_path, turn_pcm, sample_rate)
                        text = helper.transcribe_openai_windows(
                            tmp_path,
                            key=None,
                            base_url=args.base_url,
                            model="Qwen/Qwen3-ASR-1.7B",
                        ).strip()
                    finally:
                        tmp_path.unlink(missing_ok=True)
                if cached is None or cached["pcm_sha256"] != digest:
                    record = {
                        "turn_index": turn_index,
                        "samples": samples,
                        "pcm_sha256": digest,
                        "text": text,
                    }
                    cache_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    cache_handle.flush()
                    cache[turn_index] = record
                texts.append(text)

        prediction = "".join(text + "。" for text in texts if text)
        source_wav = snap / f"main_result/audio/acl6060/2022.acl-long.{talk}.wav"
        with wave.open(str(source_wav), "rb") as handle:
            source_ms = handle.getnframes() / handle.getframerate() * 1000.0
        units = [char for char in prediction if not char.isspace()]
        count = max(1, len(units))
        delays = [round((unit + 1) / count * source_ms, 3) for unit in range(len(units))]
        rows_out.append(
            {
                "index": index,
                "source": [str(source_wav)],
                "prediction": prediction,
                "delays": delays,
                "elapsed": delays,
            }
        )
        qa.append(
            {
                "talk": talk,
                "turns": len(turns),
                "cache_hits": cache_hits,
                "chars": len(units),
                "target_s": round(sum(turn_samples) / sample_rate, 3),
            }
        )
        print(json.dumps(qa[-1]), flush=True)

    with (run_dir / "instances.log").open("w", encoding="utf-8") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (run_dir / "session_qa.json").write_text(json.dumps({"per_talk": qa}, indent=2) + "\n")
    print(f"BOUNDARY_CONTROLLED_RUNDIR_DONE {run_dir}", flush=True)


if __name__ == "__main__":
    main()
