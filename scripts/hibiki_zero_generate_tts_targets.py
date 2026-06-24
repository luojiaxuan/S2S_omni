#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.hibiki_zero import HibikiSample, english_mfa_transcript
from s2s_omni.io import read_jsonl
from s2s_omni.rasst import audio_duration_s


class TransientTTSError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize English teacher target speech for Hibiki-Zero manifests."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18112/v1/audio/speech")
    parser.add_argument("--urls", default="")
    parser.add_argument("--backend", default="moss_tts_http")
    parser.add_argument("--language", default="English")
    parser.add_argument("--voice-clone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixed-ref-audio", default="")
    parser.add_argument("--ref-text", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-transient-errors", type=int, default=8)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
        handle.flush()


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["sample_id"]) for row in read_jsonl(path) if row.get("sample_id")}


def first_reference_audio(sample: HibikiSample) -> str | None:
    for chunk in sample.source_audio_chunks:
        if chunk.source_audio:
            return chunk.source_audio
    return None


def generate_one(sample: HibikiSample, args: argparse.Namespace, output_dir: Path, url: str) -> dict[str, Any]:
    out = sample.to_dict()
    wav_path = output_dir / "target_wav" / f"{sample.sample_id}.wav"
    mfa_wav = output_dir / "mfa_corpus" / f"{sample.sample_id}.wav"
    mfa_txt = output_dir / "mfa_corpus" / f"{sample.sample_id}.txt"
    try:
        if not sample.compressed_en_text:
            raise RuntimeError("empty compressed_en_text")
        payload: dict[str, Any] = {
            "input": sample.compressed_en_text,
            "language": args.language,
        }
        ref_audio = args.fixed_ref_audio or first_reference_audio(sample)
        if args.voice_clone and ref_audio:
            payload["references"] = [{"audio_path": ref_audio, "text": args.ref_text}]
        started = time.perf_counter()
        try:
            response = requests.post(url, json=payload, timeout=args.timeout_s)
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise TransientTTSError(str(exc)) from exc
        synth_s = round(time.perf_counter() - started, 6)
        if response.status_code >= 500:
            raise TransientTTSError(f"http_{response.status_code}:{response.text[:500]}")
        if response.status_code >= 400:
            raise RuntimeError(f"http_{response.status_code}:{response.text[:500]}")
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(response.content)
        target_duration_s = round(audio_duration_s(wav_path), 6)
        mfa_wav.parent.mkdir(parents=True, exist_ok=True)
        mfa_wav.write_bytes(response.content)
        mfa_txt.write_text(english_mfa_transcript(sample.compressed_en_text), encoding="utf-8")
        out.update(
            {
                "accepted": True,
                "target_en_wav": str(wav_path),
                "target_duration_s": target_duration_s,
                "mfa_wav_path": str(mfa_wav),
                "mfa_txt_path": str(mfa_txt),
                "tts_backend": args.backend,
                "tts_url": url,
                "tts_synth_s": synth_s,
                "tts_bytes": len(response.content),
                "voice_clone_reference_audio": ref_audio,
                "reject_reasons": [],
            }
        )
        return out
    except TransientTTSError as exc:
        out.update(
            {
                "accepted": False,
                "transient": True,
                "reject_reasons": [f"transient:{type(exc).__name__}"],
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "tts_backend": args.backend,
                "tts_url": url,
            }
        )
        return out
    except Exception as exc:
        out.update(
            {
                "accepted": False,
                "reject_reasons": [f"exception:{type(exc).__name__}"],
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "tts_backend": args.backend,
                "tts_url": url,
            }
        )
        return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    accepted_path = output_dir / "tts_manifest.jsonl"
    rejected_path = output_dir / "tts_rejected.jsonl"
    transient_path = output_dir / "tts_transient.jsonl"
    if not args.resume:
        for path in [accepted_path, rejected_path, transient_path]:
            if path.exists():
                path.unlink()
    urls = [part.strip() for part in args.urls.split(",") if part.strip()] or [args.url]
    done = existing_ids(accepted_path) | existing_ids(rejected_path)
    samples = []
    for idx, row in enumerate(read_jsonl(args.input)):
        if args.max_records and idx >= args.max_records:
            break
        sample = HibikiSample.from_dict(row)
        if sample.sample_id not in done:
            samples.append(sample)

    accepted = rejected = transient = processed = 0
    next_index = 0

    def submit_next(
        executor: concurrent.futures.ThreadPoolExecutor,
        pending: dict[concurrent.futures.Future[dict[str, Any]], HibikiSample],
    ) -> None:
        nonlocal next_index
        sample = samples[next_index]
        url = urls[next_index % len(urls)]
        pending[executor.submit(generate_one, sample, args, output_dir, url)] = sample
        next_index += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        pending: dict[concurrent.futures.Future[dict[str, Any]], HibikiSample] = {}
        for _ in range(min(len(samples), max(1, args.workers) * 2)):
            submit_next(executor, pending)
        while pending:
            done_futures, _ = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            stop = False
            for future in done_futures:
                sample = pending.pop(future)
                processed += 1
                try:
                    row = future.result()
                except Exception as exc:
                    row = sample.to_dict()
                    row.update(
                        {
                            "accepted": False,
                            "reject_reasons": [f"future_exception:{type(exc).__name__}"],
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                if row.get("accepted"):
                    append_jsonl(accepted_path, row)
                    accepted += 1
                elif row.get("transient"):
                    append_jsonl(transient_path, row)
                    transient += 1
                else:
                    append_jsonl(rejected_path, row)
                    rejected += 1
                if args.log_every > 0 and processed % args.log_every == 0:
                    print(
                        json.dumps(
                            {
                                "processed": processed,
                                "accepted": accepted,
                                "rejected": rejected,
                                "transient": transient,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if transient >= args.max_transient_errors:
                    stop = True
                elif next_index < len(samples):
                    submit_next(executor, pending)
            if stop:
                for future in pending:
                    future.cancel()
                pending.clear()

    summary = {
        "input": args.input,
        "output_dir": str(output_dir),
        "selected_this_run": len(samples),
        "accepted_this_run": accepted,
        "rejected_this_run": rejected,
        "transient_this_run": transient,
        "accepted_total": len(existing_ids(accepted_path)),
        "backend": args.backend,
        "urls": urls,
        "workers": args.workers,
    }
    (output_dir / "tts_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if transient:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
