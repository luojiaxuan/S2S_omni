#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import difflib
import json
import re
import shutil
import sys
import tempfile
import unicodedata
import wave
from pathlib import Path
from typing import Any

from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.openai_asr import transcribe_openai_json

CHAR_LEVEL_LANGS = {"zh", "ja"}
TIMING_METHOD = "target_speech_word_timestamp_to_pcm_packet_playout_v1"
ZH_SIMPLIFIER = OpenCC("t2s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace one ACL6060 run's text-delta timing with target-speech "
            "ASR units mapped to PCM packet playout completion."
        )
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--api-key-file", required=True, type=Path)
    parser.add_argument("--asr-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--asr-model", default="gpt-4o-mini-transcribe")
    parser.add_argument("--timestamp-model", default="whisper-1")
    parser.add_argument("--window-s", type=float, default=120.0)
    parser.add_argument("--min-alignment-coverage", type=float, default=0.6)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def read_secret(path: Path) -> str:
    value = path.expanduser().read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"empty API key file: {path}")
    return value


def wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def slice_wav(input_path: Path, output_path: Path, start_s: float, end_s: float) -> None:
    with wave.open(str(input_path), "rb") as source:
        frame_rate = source.getframerate()
        start_frame = max(0, round(start_s * frame_rate))
        end_frame = min(source.getnframes(), round(end_s * frame_rate))
        source.setpos(start_frame)
        frames = source.readframes(max(0, end_frame - start_frame))
        params = source.getparams()
    with wave.open(str(output_path), "wb") as target:
        target.setparams(params)
        target.writeframes(frames)


def fixed_window_ranges(duration_s: float, max_window_s: float) -> list[tuple[float, float]]:
    if duration_s <= 0:
        return []
    if max_window_s <= 0:
        return [(0.0, duration_s)]
    ranges: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_s:
        end = min(duration_s, start + max_window_s)
        ranges.append((start, end))
        start = end
    return ranges


def existing_kit_ranges(sample_dir: Path) -> list[tuple[float, float]]:
    rows = read_jsonl(sample_dir / "target_asr_windows.jsonl")
    ranges = [
        (float(row["start_s"]), float(row["end_s"]))
        for row in rows
        if row.get("start_s") is not None and row.get("end_s") is not None
    ]
    return ranges


def transcribe_windows(
    *,
    audio_path: Path,
    sample_dir: Path,
    api_key: str,
    base_url: str,
    asr_model: str,
    timestamp_model: str,
    target_lang: str,
    window_s: float,
    resume: bool,
) -> list[dict[str, Any]]:
    output_path = sample_dir / "target_speech_windows.jsonl"
    existing = {
        int(row["window_index"]): row
        for row in read_jsonl(output_path)
        if row.get("window_index") is not None
    }
    ranges = existing_kit_ranges(sample_dir)
    if not ranges:
        ranges = fixed_window_ranges(wav_duration_s(audio_path), window_s)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="acl6060_target_speech_") as tmp:
        tmp_dir = Path(tmp)
        for window_index, (start_s, end_s) in enumerate(ranges):
            cached = existing.get(window_index)
            if (
                resume
                and cached is not None
                and cached.get("asr_model") == asr_model
                and cached.get("timestamp_model") == timestamp_model
                and abs(float(cached.get("start_s") or 0.0) - start_s) < 0.01
                and abs(float(cached.get("end_s") or 0.0) - end_s) < 0.01
                and cached.get("timestamp_words")
            ):
                rows.append(cached)
                continue
            window_path = tmp_dir / f"window_{window_index:04d}.wav"
            slice_wav(audio_path, window_path, start_s, end_s)
            asr = transcribe_openai_json(
                api_key,
                base_url,
                asr_model,
                window_path,
                language=target_lang,
            )
            timestamped = transcribe_openai_json(
                api_key,
                base_url,
                timestamp_model,
                window_path,
                response_format="verbose_json",
                language=target_lang,
                timestamp_granularities=("word",),
            )
            words = [
                {
                    "word": str(word.get("word") or ""),
                    "start_s": round(start_s + float(word["start"]), 6),
                    "end_s": round(start_s + float(word["end"]), 6),
                }
                for word in timestamped.get("words") or []
                if word.get("start") is not None and word.get("end") is not None
            ]
            rows.append(
                {
                    "window_index": window_index,
                    "start_s": round(start_s, 6),
                    "end_s": round(end_s, 6),
                    "asr_model": asr_model,
                    "asr_text": str(asr.get("text") or "").strip(),
                    "timestamp_model": timestamp_model,
                    "timestamp_text": str(timestamped.get("text") or "").strip(),
                    "timestamp_words": words,
                }
            )
            write_jsonl(output_path, rows)
    return rows


def normalize_unit(value: str, target_lang: str | None = None) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    if target_lang == "zh":
        normalized = ZH_SIMPLIFIER.convert(normalized)
    return "".join(char for char in normalized if char.isalnum())


def text_units(text: str, target_lang: str) -> list[str]:
    if target_lang in CHAR_LEVEL_LANGS:
        return [char for char in text if not char.isspace()]
    return re.findall(r"\S+", text)


def timed_alignment_units(words: list[dict[str, Any]], target_lang: str) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for word in words:
        text = str(word["word"])
        start_ms = float(word["start_s"]) * 1000.0
        end_ms = float(word["end_s"]) * 1000.0
        pieces = (
            [char for char in text if not char.isspace()]
            if target_lang in CHAR_LEVEL_LANGS
            else [text]
        )
        if not pieces:
            continue
        duration_ms = max(0.0, end_ms - start_ms)
        for index, piece in enumerate(pieces):
            piece_start = start_ms + duration_ms * index / len(pieces)
            piece_end = start_ms + duration_ms * (index + 1) / len(pieces)
            units.append(
                {
                    "unit": piece,
                    "normalized": normalize_unit(piece, target_lang),
                    "audio_start_ms": piece_start,
                    "audio_end_ms": piece_end,
                }
            )
    return units


def align_hypothesis_units(
    hypothesis: str,
    target_lang: str,
    timed_units: list[dict[str, Any]],
) -> tuple[list[float], dict[str, Any]]:
    units = text_units(hypothesis, target_lang)
    hyp_lexical = [
        (index, normalized)
        for index, unit in enumerate(units)
        if (normalized := normalize_unit(unit, target_lang))
    ]
    timed_lexical = [
        (index, str(unit["normalized"]))
        for index, unit in enumerate(timed_units)
        if unit["normalized"]
    ]
    matcher = difflib.SequenceMatcher(
        None,
        [value for _, value in hyp_lexical],
        [value for _, value in timed_lexical],
        autojunk=False,
    )
    mapped: dict[int, float] = {}
    matched_lexical = 0
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            hyp_index = hyp_lexical[block.a + offset][0]
            timed_index = timed_lexical[block.b + offset][0]
            mapped[hyp_index] = float(timed_units[timed_index]["audio_end_ms"])
            matched_lexical += 1
    if not mapped:
        raise ValueError("target-speech ASR and timestamp transcript have no aligned units")

    known = sorted(mapped)
    audio_end_ms: list[float] = []
    for index in range(len(units)):
        if index in mapped:
            audio_end_ms.append(mapped[index])
            continue
        position = bisect.bisect_left(known, index)
        left = known[position - 1] if position > 0 else None
        right = known[position] if position < len(known) else None
        if left is None:
            value = mapped[right]  # type: ignore[index]
        elif right is None:
            value = mapped[left]
        else:
            ratio = (index - left) / (right - left)
            value = mapped[left] + (mapped[right] - mapped[left]) * ratio
        audio_end_ms.append(value)
    for index in range(1, len(audio_end_ms)):
        audio_end_ms[index] = max(audio_end_ms[index], audio_end_ms[index - 1])
    lexical_count = len(hyp_lexical)
    return audio_end_ms, {
        "hypothesis_units": len(units),
        "hypothesis_lexical_units": lexical_count,
        "timestamp_units": len(timed_units),
        "matched_lexical_units": matched_lexical,
        "alignment_coverage": matched_lexical / lexical_count if lexical_count else 0.0,
    }


def packet_playout_timeline(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    playout_end_ms = 0.0
    for index, packet in enumerate(packets):
        arrival_ms = float(packet["received_at_ms"])
        duration_ms = float(packet["duration_ms"])
        playout_start_ms = max(arrival_ms, playout_end_ms)
        playout_end_ms = playout_start_ms + duration_ms
        output.append(
            {
                **packet,
                "packet_index": index,
                "playout_start_ms": round(playout_start_ms, 3),
                "playout_end_ms": round(playout_end_ms, 3),
                "queue_delay_ms": round(playout_start_ms - arrival_ms, 3),
            }
        )
    return output


def kit_packets(sample_dir: Path, source_length_ms: float) -> list[dict[str, Any]]:
    rows = read_jsonl(sample_dir / "audio_chunks.jsonl")
    output: list[dict[str, Any]] = []
    audio_start_ms = 0.0
    for row in rows:
        if row.get("arrival_s") is None:
            continue
        duration_ms = float(row.get("audio_duration_s") or 0.0) * 1000.0
        if duration_ms <= 0:
            continue
        arrival_ms = float(row["arrival_s"]) * 1000.0
        output.append(
            {
                "packet_index": len(output),
                "received_at_ms": round(arrival_ms, 3),
                "sent_source_ms": round(min(source_length_ms, arrival_ms), 3),
                "sample_rate": 16000,
                "sample_count": round(duration_ms * 16.0),
                "duration_ms": round(duration_ms, 3),
                "audio_start_ms": round(audio_start_ms, 3),
                "audio_end_ms": round(audio_start_ms + duration_ms, 3),
            }
        )
        audio_start_ms += duration_ms
    return packet_playout_timeline(output)


def packet_for_audio_position(packets: list[dict[str, Any]], audio_end_ms: float) -> dict[str, Any]:
    ends = [float(packet["audio_end_ms"]) for packet in packets]
    index = min(len(packets) - 1, bisect.bisect_left(ends, audio_end_ms))
    return packets[index]


def unit_playout_times(
    audio_end_ms: list[float],
    packets: list[dict[str, Any]],
    source_length_ms: float,
) -> tuple[list[float], list[float]]:
    elapsed: list[float] = []
    delays: list[float] = []
    for position_ms in audio_end_ms:
        packet = packet_for_audio_position(packets, position_ms)
        within_packet_ms = min(
            float(packet["duration_ms"]),
            max(0.0, position_ms - float(packet["audio_start_ms"])),
        )
        playout_ms = float(packet["playout_start_ms"]) + within_packet_ms
        elapsed.append(round(playout_ms, 3))
        sent_source_ms = packet.get("sent_source_ms")
        delays.append(
            round(
                min(
                    source_length_ms,
                    float(sent_source_ms)
                    if sent_source_ms is not None
                    else float(packet["received_at_ms"]),
                ),
                3,
            )
        )
    return delays, elapsed


def locate_audio(response: dict[str, Any], sample_dir: Path) -> Path:
    configured = str(response.get("target_audio") or response.get("target_wav") or "")
    candidates = [
        Path(configured) if configured else None,
        sample_dir / "target_tts.wav",
        *sorted(sample_dir.glob("target_audio_*.wav")),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    raise FileNotFoundError(f"target audio not found for {sample_dir}")


def locate_packets(
    response: dict[str, Any],
    sample_dir: Path,
    provider: str,
    source_length_ms: float,
) -> list[dict[str, Any]]:
    if provider == "kit":
        packets = kit_packets(sample_dir, source_length_ms)
    else:
        configured = str(response.get("target_audio_packets") or "")
        path = Path(configured) if configured else sample_dir / "target_audio_packets.jsonl"
        packets = read_jsonl(path)
    if not packets:
        raise ValueError(f"target audio packet timeline is empty: {sample_dir}")
    return packet_playout_timeline(packets)


def build_instances(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "run_config.json"
    config = read_json(config_path)
    provider = str(config["provider"])
    target_lang = str(config["target_lang"])
    api_key = read_secret(args.api_key_file)
    provider_instances_path = run_dir / "instances.provider_transcript.log"
    instances_path = run_dir / "instances.log"
    if not provider_instances_path.exists():
        shutil.copy2(instances_path, provider_instances_path)
    source_rows = {int(row["index"]): row for row in read_jsonl(provider_instances_path)}
    responses = read_jsonl(run_dir / "responses.jsonl")
    output_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for response in responses:
        index = int(response["index"])
        source = source_rows[index]
        source_length_ms = float(source["source_length"])
        sample_dir = Path(str(response["run_dir"]))
        audio_path = locate_audio(response, sample_dir)
        packets = locate_packets(
            response,
            sample_dir,
            provider,
            source_length_ms,
        )
        windows = transcribe_windows(
            audio_path=audio_path,
            sample_dir=sample_dir,
            api_key=api_key,
            base_url=args.asr_base_url,
            asr_model=args.asr_model,
            timestamp_model=args.timestamp_model,
            target_lang=target_lang,
            window_s=args.window_s,
            resume=args.resume,
        )
        prediction = " ".join(
            str(row.get("asr_text") or "").strip()
            for row in windows
            if str(row.get("asr_text") or "").strip()
        )
        timestamp_words = [word for row in windows for word in row.get("timestamp_words") or []]
        timed_units = timed_alignment_units(timestamp_words, target_lang)
        audio_unit_ends, alignment = align_hypothesis_units(
            prediction,
            target_lang,
            timed_units,
        )
        if alignment["alignment_coverage"] < args.min_alignment_coverage:
            raise ValueError(
                f"target-speech alignment coverage too low for {sample_dir}: "
                f"{alignment['alignment_coverage']:.3f}"
            )
        delays, elapsed = unit_playout_times(
            audio_unit_ends,
            packets,
            source_length_ms,
        )
        output_rows.append(
            {
                "index": index,
                "prediction": prediction,
                "delays": delays,
                "elapsed": elapsed,
                "prediction_length": len(elapsed),
                "reference": source["reference"],
                "source": source["source"],
                "source_length": source_length_ms,
                "timing_method": TIMING_METHOD,
            }
        )
        timing_rows.append(
            {
                "index": index,
                "run_id": response["run_id"],
                "provider": provider,
                "target_audio": str(audio_path),
                "target_audio_duration_ms": round(wav_duration_s(audio_path) * 1000.0, 3),
                "target_audio_packet_count": len(packets),
                "target_audio_last_arrival_ms": packets[-1]["received_at_ms"],
                "target_audio_playout_end_ms": packets[-1]["playout_end_ms"],
                "target_speech_last_unit_playout_ms": elapsed[-1],
                "target_speech_asr_model": args.asr_model,
                "target_speech_timestamp_model": args.timestamp_model,
                "timing_method": TIMING_METHOD,
                **alignment,
            }
        )
    write_jsonl(instances_path, output_rows)
    write_jsonl(run_dir / "target_speech_timing.jsonl", timing_rows)
    config.update(
        {
            "candidate_text_source": "target_speech_asr_gpt4o_mini_transcribe",
            "target_speech_asr_model": args.asr_model,
            "target_speech_timestamp_model": args.timestamp_model,
            "latency_timing_method": TIMING_METHOD,
            "target_speech_playout_assumption": "zero_jitter_immediate_pcm_playout",
        }
    )
    write_json(config_path, config)
    summary = {
        "run_dir": str(run_dir),
        "provider": provider,
        "target_lang": target_lang,
        "samples": len(output_rows),
        "timing_method": TIMING_METHOD,
        "alignment_coverage_min": min(float(row["alignment_coverage"]) for row in timing_rows),
        "alignment_coverage_mean": sum(float(row["alignment_coverage"]) for row in timing_rows)
        / len(timing_rows),
    }
    write_json(run_dir / "target_speech_timing_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(build_instances(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
