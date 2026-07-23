#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KIT_SAMPLE_RATE = 16000
TARGET_LANGUAGES = {"zh": "Chinese", "de": "German", "ja": "Japanese"}
CHAR_LEVEL_LANGS = {"zh", "ja"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ACL6060 full-wav live S2S through KIT Lecture Translator and "
            "write an instances.log from target-speech ASR."
        )
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cookie-header-file", required=True, type=Path)
    parser.add_argument("--api-key-file", required=True, type=Path)
    parser.add_argument("--target-lang", required=True, choices=sorted(TARGET_LANGUAGES))
    parser.add_argument("--chunk-ms", type=int, default=960)
    parser.add_argument("--speed-factor", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-audio-seconds", type=float, default=0.0)
    parser.add_argument("--base-url", default="https://lt2srv.iar.kit.edu")
    parser.add_argument("--asr-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--asr-model", default="gpt-4o-mini-transcribe")
    parser.add_argument("--asr-window-s", type=float, default=120.0)
    parser.add_argument("--tts-quality-mode", default="high_quality")
    parser.add_argument("--format", default="mixed", choices=["online", "mixed"])
    parser.add_argument("--language-order", default="target,en", choices=["target,en", "en,target"])
    parser.add_argument("--availability", default="private")
    parser.add_argument("--smart-chaptering", default="online_dynamic")
    parser.add_argument("--pace", default="realtime", choices=["realtime", "accelerated"])
    parser.add_argument("--accelerated-sleep-s", type=float, default=0.12)
    parser.add_argument("--poll-after-s", type=float, default=180.0)
    parser.add_argument("--poll-interval-s", type=float, default=20.0)
    parser.add_argument("--download-hf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--create-script", type=Path, default=ROOT / "scripts/create_kit_session.py")
    parser.add_argument("--stream-script", type=Path, default=ROOT / "scripts/run_kit_live_floras.py")
    parser.add_argument("--fetch-script", type=Path, default=ROOT / "scripts/fetch_kit_target_audio.py")
    return parser.parse_args()


def load_acl_live_module() -> Any:
    path = ROOT / "projects/acl6060_s2s_metrics_seed/run_acl6060_live_stream_eval.py"
    spec = importlib.util.spec_from_file_location("acl6060_live_stream_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_secret(path: Path) -> str:
    value = path.expanduser().read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"empty secret file: {path}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=False) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def load_done_indices(instances_path: Path) -> set[int]:
    done: set[int] = set()
    for row in read_jsonl(instances_path):
        done.add(int(row["index"]))
    return done


def select_indices(total: int, start_index: int, limit: int) -> list[int]:
    end = total if limit <= 0 else min(total, start_index + limit)
    return list(range(start_index, end))


def speed_tag(speed: float) -> str:
    return ("%g" % speed).replace(".", "p")


def run_cmd(cmd: list[str]) -> None:
    print(json.dumps({"cmd": cmd}, ensure_ascii=False), flush=True)
    subprocess.run(cmd, check=True)


def target_languages(args: argparse.Namespace) -> list[str]:
    if args.language_order == "en,target":
        return ["en", args.target_lang]
    return [args.target_lang, "en"]


def session_name(args: argparse.Namespace, index: int, run_id: str) -> str:
    return (
        f"acl6060_en{args.target_lang}_kit_{args.format}_hq_"
        f"chunk{args.chunk_ms}_speed{speed_tag(args.speed_factor)}_{index:03d}_{run_id}"
    )


def create_session(args: argparse.Namespace, out_path: Path, name: str) -> dict[str, Any]:
    if args.resume and out_path.exists():
        existing = read_json(out_path)
        if existing.get("session_id"):
            return existing
    cmd = [
        args.python_bin,
        str(args.create_script),
        "--cookie-header-file",
        str(args.cookie_header_file),
        "--output-json",
        str(out_path),
        "--base-url",
        args.base_url,
        "--name",
        name,
        "--mt-language",
        args.target_lang,
        "--audio-language",
        args.target_lang,
        "--tts-quality-mode",
        args.tts_quality_mode,
        "--format",
        args.format,
        "--availability",
        args.availability,
        "--smart-chaptering",
        args.smart_chaptering,
    ]
    for language in target_languages(args):
        cmd.extend(["--language", language])
    run_cmd(cmd)
    created = read_json(out_path)
    if not created.get("session_id"):
        raise RuntimeError(f"KIT create did not return a session_id: {out_path}")
    return created


def run_stream(args: argparse.Namespace, run_json: Path, stream_wav: Path, created: dict[str, Any], name: str) -> None:
    if args.resume and run_json.exists():
        existing = read_json(run_json)
        if existing.get("pauseStatus", {}).get("ok") and existing.get("collection"):
            return
    cmd = [
        args.python_bin,
        str(args.stream_script),
        "--session-id",
        str(created["session_id"]),
        "--session-name",
        name,
        "--wav-path",
        str(stream_wav),
        "--output-json",
        str(run_json),
        "--cookie-header-file",
        str(args.cookie_header_file),
        "--base-url",
        args.base_url,
        "--chunk-s",
        f"{args.chunk_ms / 1000.0:.6f}",
        "--pace",
        args.pace,
        "--accelerated-sleep-s",
        str(args.accelerated_sleep_s),
        "--poll-after-s",
        str(args.poll_after_s),
        "--poll-interval-s",
        str(args.poll_interval_s),
    ]
    run_cmd(cmd)


def fetch_target_audio(args: argparse.Namespace, run_json: Path, target_wav: Path, chunks_jsonl: Path) -> None:
    if args.resume and target_wav.exists() and chunks_jsonl.exists():
        return
    cmd = [
        args.python_bin,
        str(args.fetch_script),
        "--run-json",
        str(run_json),
        "--cookie-header-file",
        str(args.cookie_header_file),
        "--output-wav",
        str(target_wav),
        "--output-chunks-jsonl",
        str(chunks_jsonl),
        "--base-url",
        args.base_url,
    ]
    run_cmd(cmd)


def wav_duration_ms(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return round(handle.getnframes() * 1000.0 / float(handle.getframerate()), 3)


def grouped_asr_window_ranges(
    chunks_jsonl: Path,
    target_duration_s: float,
    max_window_s: float,
) -> list[tuple[float, float]]:
    if max_window_s <= 0:
        return [(0.0, target_duration_s)]
    chunks = chunk_rows_for_timing(chunks_jsonl)
    if not chunks:
        ranges = []
        start = 0.0
        while start < target_duration_s:
            end = min(target_duration_s, start + max_window_s)
            ranges.append((start, end))
            start = end
        return ranges
    ranges: list[tuple[float, float]] = []
    window_start = 0.0
    previous_end = 0.0
    for row in chunks:
        chunk_end = float(row.get("cumulative_audio_duration_s") or 0.0)
        if chunk_end <= previous_end:
            chunk_end = previous_end + float(row.get("audio_duration_s") or 0.0)
        chunk_end = min(target_duration_s, chunk_end)
        if previous_end > window_start and chunk_end - window_start > max_window_s:
            ranges.append((window_start, previous_end))
            window_start = previous_end
        previous_end = chunk_end
    final_end = target_duration_s
    if final_end > window_start:
        ranges.append((window_start, final_end))
    return ranges


def slice_wav(input_path: Path, output_path: Path, start_s: float, end_s: float) -> None:
    with wave.open(str(input_path), "rb") as source:
        frame_rate = source.getframerate()
        start_frame = max(0, int(round(start_s * frame_rate)))
        end_frame = min(source.getnframes(), int(round(end_s * frame_rate)))
        source.setpos(start_frame)
        frames = source.readframes(max(0, end_frame - start_frame))
        params = source.getparams()
    with wave.open(str(output_path), "wb") as target:
        target.setparams(params)
        target.writeframes(frames)


def transcribe_target(
    args: argparse.Namespace,
    api_key: str,
    target_wav: Path,
    chunks_jsonl: Path,
    asr_json: Path,
    asr_windows_jsonl: Path,
) -> dict[str, Any]:
    strategy = "tts_chunk_grouped_windows_v1"
    if args.resume and asr_json.exists():
        existing = read_json(asr_json)
        if (
            str(existing.get("asr_text") or "").strip()
            and existing.get("asr_strategy") == strategy
            and existing.get("asr_model") == args.asr_model
            and float(existing.get("asr_window_s") or 0.0) == args.asr_window_s
        ):
            return existing
    from s2s_omni.openai_asr import transcode_for_upload, transcribe_openai

    target_duration_s = wav_duration_ms(target_wav) / 1000.0
    ranges = grouped_asr_window_ranges(
        chunks_jsonl,
        target_duration_s,
        args.asr_window_s,
    )
    existing_windows = {
        int(row["window_index"]): row
        for row in read_jsonl(asr_windows_jsonl)
        if row.get("window_index") is not None
    }
    window_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="acl6060_kit_asr_") as tmp:
        tmp_dir = Path(tmp)
        for window_index, (start_s, end_s) in enumerate(ranges):
            existing = existing_windows.get(window_index)
            if (
                existing is not None
                and abs(float(existing.get("start_s") or 0.0) - start_s) < 0.01
                and abs(float(existing.get("end_s") or 0.0) - end_s) < 0.01
                and existing.get("asr_model") == args.asr_model
            ):
                window_rows.append(existing)
                continue
            window_wav = tmp_dir / f"window_{window_index:04d}.wav"
            slice_wav(target_wav, window_wav, start_s, end_s)
            upload = transcode_for_upload(window_wav, tmp_dir, 25 * 1024 * 1024)
            text = transcribe_openai(
                api_key,
                args.asr_base_url,
                args.asr_model,
                upload,
            )
            window_rows.append(
                {
                    "window_index": window_index,
                    "start_s": round(start_s, 3),
                    "end_s": round(end_s, 3),
                    "duration_s": round(end_s - start_s, 3),
                    "asr_text": text,
                    "asr_model": args.asr_model,
                    "uploaded_audio_size_bytes": upload.stat().st_size,
                }
            )
            write_jsonl(asr_windows_jsonl, window_rows)
    text = " ".join(
        str(row.get("asr_text") or "").strip()
        for row in window_rows
        if str(row.get("asr_text") or "").strip()
    )
    payload = {
        "asr_text": text,
        "asr_model": args.asr_model,
        "asr_strategy": strategy,
        "asr_window_s": args.asr_window_s,
        "asr_window_count": len(window_rows),
        "asr_windows_jsonl": str(asr_windows_jsonl),
        "target_wav": str(target_wav),
        "target_wav_bytes": target_wav.stat().st_size,
    }
    write_json(asr_json, payload)
    return payload


def text_units(text: str, target_lang: str) -> list[str]:
    if target_lang in CHAR_LEVEL_LANGS:
        return [ch for ch in text if not ch.isspace()]
    return text.split()


def chunk_rows_for_timing(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        if row.get("arrival_s") is None:
            continue
        try:
            duration = float(row.get("audio_duration_s") or 0.0)
            arrival = float(row["arrival_s"])
        except (TypeError, ValueError):
            continue
        if duration <= 0.0:
            continue
        out = dict(row)
        out["audio_duration_s"] = duration
        out["arrival_s"] = max(0.0, arrival)
        out["cumulative_audio_duration_s"] = float(row.get("cumulative_audio_duration_s") or 0.0)
        rows.append(out)
    return rows


def target_unit_times_ms(
    *,
    text: str,
    target_lang: str,
    chunks_jsonl: Path,
    source_length_ms: float,
) -> tuple[list[float], list[float], dict[str, Any]]:
    units = text_units(text, target_lang)
    if not units:
        return [], [], {"timing_method": "empty_asr_text", "units": 0}

    chunks = chunk_rows_for_timing(chunks_jsonl)
    if not chunks:
        fallback = [source_length_ms] * len(units)
        return fallback, fallback, {"timing_method": "source_end_fallback", "units": len(units)}

    total_audio_s = max(
        chunks[-1].get("cumulative_audio_duration_s") or 0.0,
        sum(float(row.get("audio_duration_s") or 0.0) for row in chunks),
    )
    if total_audio_s <= 0.0:
        total_audio_s = sum(float(row.get("audio_duration_s") or 0.0) for row in chunks)
    cursor = 0
    elapsed: list[float] = []
    delays: list[float] = []
    prev_elapsed = 0.0
    for unit_index in range(len(units)):
        target_audio_pos_s = total_audio_s * (unit_index + 0.5) / len(units)
        while (
            cursor + 1 < len(chunks)
            and float(chunks[cursor].get("cumulative_audio_duration_s") or 0.0) < target_audio_pos_s
        ):
            cursor += 1
        elapsed_ms = max(prev_elapsed, float(chunks[cursor]["arrival_s"]) * 1000.0)
        prev_elapsed = elapsed_ms
        elapsed.append(round(elapsed_ms, 3))
        delays.append(round(min(source_length_ms, elapsed_ms), 3))
    return delays, elapsed, {
        "timing_method": "target_audio_chunk_arrival_proportional_units",
        "units": len(units),
        "tts_audio_chunks": len(chunks),
        "target_audio_duration_ms": round(total_audio_s * 1000.0, 3),
        "last_arrival_ms": round(float(chunks[-1]["arrival_s"]) * 1000.0, 3),
    }


def sample_result_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "create_result": run_dir / "create_result.json",
        "run_json": run_dir / "run.json",
        "target_wav": run_dir / "target_tts.wav",
        "chunks_jsonl": run_dir / "audio_chunks.jsonl",
        "asr_json": run_dir / "target_asr.json",
        "asr_windows_jsonl": run_dir / "target_asr_windows.jsonl",
    }


def main() -> None:
    args = parse_args()
    acl = load_acl_live_module()
    if args.download_hf:
        acl.download_hf_subset(args.dataset_root, args.target_lang)
    paths = acl.resolve_paths(args.dataset_root, args.target_lang)
    source_lines = acl.read_lines(paths.source_list)
    target_lines = acl.read_lines(paths.target_list)
    audio_rows = acl.parse_simple_audio_yaml(paths.audio_yaml)
    if len(source_lines) != len(target_lines):
        raise ValueError("source/target line count mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    instances_path = args.output_dir / "instances.log"
    responses_path = args.output_dir / "responses.jsonl"
    if not args.resume:
        instances_path.write_text("", encoding="utf-8")
        responses_path.write_text("", encoding="utf-8")
    done = load_done_indices(instances_path) if args.resume else set()
    api_key = read_secret(args.api_key_file)
    run_config = {
        "provider": "kit",
        "model": "KIT Lecture Translator",
        "dataset_root": str(args.dataset_root),
        "source_list": str(paths.source_list),
        "target_list": str(paths.target_list),
        "ref_file": str(paths.ref_file),
        "source_text_file": str(paths.source_text_file),
        "audio_yaml": str(paths.audio_yaml),
        "audio_yaml_rows": len(audio_rows),
        "chunk_ms": args.chunk_ms,
        "speed_factor": args.speed_factor,
        "target_lang": args.target_lang,
        "kit_base_url": args.base_url,
        "kit_languages": target_languages(args),
        "kit_mt_language": args.target_lang,
        "kit_audio_language": args.target_lang,
        "kit_format": args.format,
        "kit_tts_quality_mode": args.tts_quality_mode,
        "kit_availability": args.availability,
        "kit_smart_chaptering": args.smart_chaptering,
        "asr_model": args.asr_model,
        "asr_base_url": args.asr_base_url,
        "asr_strategy": "tts_chunk_grouped_windows_v1",
        "asr_window_s": args.asr_window_s,
        "candidate_text_source": "target_speech_asr_gpt4o_mini_transcribe",
    }
    write_json(args.output_dir / "run_config.json", run_config)

    for index in select_indices(len(source_lines), args.start_index, args.limit):
        if index in done:
            print(json.dumps({"skip_done": index}, ensure_ascii=False), flush=True)
            continue
        source_wav = acl.resolve_audio_path(source_lines[index], args.dataset_root, paths)
        run_id = source_wav.stem
        run_dir = args.output_dir / f"{index:03d}_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        stream_wav = run_dir / f"source_stream_{KIT_SAMPLE_RATE}_speed{speed_tag(args.speed_factor)}.wav"
        acl.ffmpeg_convert(
            source_wav,
            stream_wav,
            KIT_SAMPLE_RATE,
            args.max_audio_seconds,
            args.speed_factor,
        )
        source_length_ms = wav_duration_ms(stream_wav)
        result_paths = sample_result_paths(run_dir)
        name = session_name(args, index, run_id)
        start_s = time.monotonic()
        created = create_session(args, result_paths["create_result"], name)
        run_stream(args, result_paths["run_json"], stream_wav, created, name)
        fetch_target_audio(args, result_paths["run_json"], result_paths["target_wav"], result_paths["chunks_jsonl"])
        asr = transcribe_target(
            args,
            api_key,
            result_paths["target_wav"],
            result_paths["chunks_jsonl"],
            result_paths["asr_json"],
            result_paths["asr_windows_jsonl"],
        )
        prediction = str(asr.get("asr_text") or "").strip()
        delays, elapsed, timing = target_unit_times_ms(
            text=prediction,
            target_lang=args.target_lang,
            chunks_jsonl=result_paths["chunks_jsonl"],
            source_length_ms=source_length_ms,
        )
        if not delays:
            raise RuntimeError(f"empty KIT target ASR for index {index}: {result_paths['target_wav']}")
        elapsed_ms = (time.monotonic() - start_s) * 1000.0
        record = {
            "index": index,
            "prediction": prediction,
            "delays": delays,
            "elapsed": elapsed,
            "prediction_length": len(delays),
            "reference": target_lines[index],
            "source": [str(source_wav)],
            "source_length": source_length_ms,
        }
        append_jsonl(instances_path, record)
        response = {
            "index": index,
            "run_id": run_id,
            "provider": "kit",
            "model": "KIT Lecture Translator",
            "kit_session_id": created.get("session_id"),
            "kit_session_url": created.get("final_url"),
            "prediction_units": len(delays),
            "elapsed_ms": round(elapsed_ms, 3),
            "source_length_ms": source_length_ms,
            "target_wav": str(result_paths["target_wav"]),
            "target_asr_json": str(result_paths["asr_json"]),
            "run_dir": str(run_dir),
            **timing,
        }
        append_jsonl(responses_path, response)
        print(
            json.dumps(
                {
                    "index": index,
                    "run_id": run_id,
                    "prediction_units": len(delays),
                    "target_audio_chunks": timing.get("tts_audio_chunks"),
                    "elapsed_ms": round(elapsed_ms, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
