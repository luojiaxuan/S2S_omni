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

from s2s_omni.rasst import (
    DEFAULT_RASST_DEV,
    DEFAULT_RASST_TRAIN,
    audio_duration_s,
    iter_rasst_rows,
    sanitize_id,
)
from s2s_omni.textgrid import transcript_for_mfa


class TransientTTSError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full-sentence target wavs through an OpenAI-compatible TTS HTTP server."
    )
    parser.add_argument("--input", default=DEFAULT_RASST_TRAIN)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18112/v1/audio/speech")
    parser.add_argument("--urls", default="", help="Comma-separated TTS endpoint list. Overrides --url when set.")
    parser.add_argument("--backend", default="qwen3_tts_http")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--max-transient-errors", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--ref-text", default="")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-rejected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-mfa-corpus", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--dev", action="store_true", help=f"Shortcut for --input {DEFAULT_RASST_DEV}")
    return parser.parse_args()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")
        handle.flush()


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("row_id"):
                out.add(str(row["row_id"]))
    return out


def first_reference_audio(row: Any) -> str:
    for turn in row.turns:
        if turn.audio_path:
            return turn.audio_path
    raise RuntimeError("row has no source audio reference")


def row_record_template(row: Any, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "row_id": row.row_id,
        "source_path": row.source_path,
        "row_index": row.index,
        "full_target_text": row.full_target_text,
        "target_char_spans": [list(span) for span in row.target_char_spans],
        "chunks": [
            {
                "chunk_index": turn.index,
                "target_text": turn.assistant_text,
                "source_audio": turn.audio_path,
                "char_span": list(row.target_char_spans[turn.index]),
            }
            for turn in row.turns
        ],
        "backend": args.backend,
        "language": args.language,
        "target_codes_path": None,
        "codec_frames": None,
        "codec_num_quantizers": None,
    }


def generate_one(row: Any, args: argparse.Namespace, output_dir: Path, url: str) -> dict[str, Any]:
    safe_id = sanitize_id(row.row_id)
    out_record = row_record_template(row, args)
    wav_path = output_dir / "target_wav" / f"{safe_id}.wav"
    mfa_wav = output_dir / "mfa_corpus" / f"{safe_id}.wav"
    mfa_txt = output_dir / "mfa_corpus" / f"{safe_id}.txt"
    try:
        if not row.full_target_text:
            raise RuntimeError("empty full target text")
        ref_audio = first_reference_audio(row)
        payload: dict[str, Any] = {
            "input": row.full_target_text,
            "language": args.language,
            "max_new_tokens": args.max_new_tokens,
            "references": [{"audio_path": ref_audio, "text": args.ref_text}],
        }
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
        duration_s = round(audio_duration_s(wav_path), 6)
        out_record.update(
            {
                "accepted": True,
                "target_wav_path": str(wav_path),
                "target_duration_s": duration_s,
                "reference_audio": ref_audio,
                "synth_s": synth_s,
                "bytes": len(response.content),
                "tts_url": url,
                "reject_reasons": [],
            }
        )
        if args.write_mfa_corpus:
            mfa_wav.parent.mkdir(parents=True, exist_ok=True)
            mfa_wav.write_bytes(response.content)
            mfa_txt.write_text(transcript_for_mfa(row.full_target_text), encoding="utf-8")
            out_record["mfa_wav_path"] = str(mfa_wav)
            out_record["mfa_txt_path"] = str(mfa_txt)
        return out_record
    except TransientTTSError as exc:
        out_record.update(
            {
                "accepted": False,
                "transient": True,
                "tts_url": url,
                "reject_reasons": [f"transient:{type(exc).__name__}"],
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return out_record
    except Exception as exc:
        out_record.update(
            {
                "accepted": False,
                "reject_reasons": [f"exception:{type(exc).__name__}"],
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return out_record


def main() -> None:
    args = parse_args()
    if args.dev:
        args.input = DEFAULT_RASST_DEV
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    output_dir = Path(args.output_dir)
    urls = [part.strip() for part in args.urls.split(",") if part.strip()]
    if not urls:
        urls = [args.url]
    accepted_path = output_dir / "target_manifest.jsonl"
    rejected_path = output_dir / "target_rejected.jsonl"
    transient_path = output_dir / "target_transient.jsonl"
    if not args.resume:
        for path in [accepted_path, rejected_path, transient_path]:
            if path.exists():
                path.unlink()
    done = existing_ids(accepted_path) | existing_ids(rejected_path)

    selected = []
    eligible_index = 0
    for row in iter_rasst_rows(args.input, max_records=args.max_records):
        if row.row_id in done:
            continue
        if not row.full_target_text:
            append_jsonl(
                rejected_path,
                {
                    "row_id": row.row_id,
                    "accepted": False,
                    "reject_reasons": ["empty_full_target_text"],
                    "source_path": row.source_path,
                },
            )
            continue
        if eligible_index % args.num_shards == args.shard_index:
            selected.append(row)
        eligible_index += 1

    if not selected:
        print(json.dumps({"selected": 0, "done": len(done)}, ensure_ascii=False, indent=2))
        return

    accepted = 0
    rejected = 0
    transient = 0
    processed = 0
    next_selected_index = 0

    def submit_next(
        executor: concurrent.futures.ThreadPoolExecutor,
        pending: dict[concurrent.futures.Future[dict[str, Any]], tuple[Any, int]],
    ) -> None:
        nonlocal next_selected_index
        row = selected[next_selected_index]
        task_index = next_selected_index + 1
        url = urls[(task_index - 1) % len(urls)]
        pending[executor.submit(generate_one, row, args, output_dir, url)] = (row, task_index)
        next_selected_index += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        pending: dict[concurrent.futures.Future[dict[str, Any]], tuple[Any, int]] = {}
        for _ in range(min(len(selected), max(1, args.workers) * 2)):
            submit_next(executor, pending)
        while pending:
            done, _ = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            stop = False
            for future in done:
                row, task_index = pending.pop(future)
                processed += 1
                index = processed

                try:
                    record = future.result()
                except Exception as exc:
                    record = row_record_template(row, args)
                    record.update(
                        {
                            "accepted": False,
                            "reject_reasons": [f"future_exception:{type(exc).__name__}"],
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                if record.get("accepted"):
                    append_jsonl(accepted_path, record)
                    accepted += 1
                elif record.get("transient"):
                    append_jsonl(transient_path, record)
                    transient += 1
                else:
                    rejected += 1
                    if args.save_rejected:
                        append_jsonl(rejected_path, record)
                if args.log_every > 0 and index % args.log_every == 0:
                    print(
                        json.dumps(
                            {
                                "processed": index,
                                "accepted": accepted,
                                "rejected": rejected,
                                "transient": transient,
                                "last_row_id": row.row_id,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if transient >= args.max_transient_errors:
                    stop = True
                elif next_selected_index < len(selected):
                    submit_next(executor, pending)
            if stop:
                for pending_future in pending:
                    pending_future.cancel()
                pending.clear()
                break

    summary = {
        "input": args.input,
        "output_dir": str(output_dir),
        "url": args.url,
        "urls": urls,
        "selected_this_run": len(selected),
        "accepted_this_run": accepted,
        "rejected_this_run": rejected,
        "transient_this_run": transient,
        "accepted_total": len(existing_ids(accepted_path)),
        "rejected_total": len(existing_ids(rejected_path)),
        "transient_total": len(existing_ids(transient_path)),
        "backend": args.backend,
        "workers": args.workers,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    (output_dir / f"summary_shard{args.shard_index}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if transient:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
