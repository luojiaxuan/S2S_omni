#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.floras_live import (
    SOURCE_WINDOW_S,
    append_jsonl,
    audio_duration_s,
    chunk_text_by_units,
    corpus_metrics,
    coverage_status,
    esc,
    read_run_manifest,
    rel,
    sanitize_id,
    slice_wav,
    summarize_numeric,
    write_combined_dashboard,
    write_run_dashboard,
)
from s2s_omni.io import read_jsonl, write_jsonl
from s2s_omni.llm_client import ChatClient, extract_json_object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FLORAS live S2S outputs.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asr-jsonl", default="")
    parser.add_argument("--window-asr-jsonl", default="")
    parser.add_argument("--target-mfa-jsonl", default="")
    parser.add_argument("--asr-model", default="")
    parser.add_argument("--asr-device", default="cuda:0")
    parser.add_argument("--coverage-judge", choices=["none", "openai"], default="none")
    parser.add_argument("--coverage-model", default="gpt-5-mini")
    parser.add_argument("--openai-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--window-s", type=float, default=SOURCE_WINDOW_S)
    parser.add_argument("--backlog-threshold-s", type=float, default=0.5)
    parser.add_argument("--target-context-s", type=float, default=20.0)
    parser.add_argument("--max-windows", type=int, default=0)
    return parser.parse_args()


def result_paths(run_output_dir: Path) -> list[Path]:
    return sorted(run_output_dir.glob("*/result.json"))


def load_results(run_output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in result_paths(run_output_dir):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["run_result_path"] = str(path)
        row["live_run_dir"] = str(path.parent)
        manifest_path = path.parent / "run_manifest.json"
        if manifest_path.exists():
            row.update(json.loads(manifest_path.read_text(encoding="utf-8")))
        rows.append(row)
    return rows


def keyed(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row.get(key)): row for row in rows if row.get(key) is not None}


def make_asr_pipe(model_name: str, device: str):
    import torch
    from transformers import pipeline

    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    return pipeline("automatic-speech-recognition", model=model_name, torch_dtype=dtype, device=device)


def transcribe(pipe: Any, wav_path: str | Path, lang: str) -> str:
    result = pipe(
        str(wav_path),
        generate_kwargs={"language": lang, "task": "transcribe"},
        return_timestamps=False,
    )
    if isinstance(result, dict):
        return str(result.get("text") or "").strip()
    return str(result).strip()


def text_span_by_ratio(text: str, lang: str, start_ratio: float, end_ratio: float) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    chunks = chunk_text_by_units(text, lang, max_units=1)
    if not chunks:
        return text
    n = len(chunks)
    start = max(0, min(n, int(round(start_ratio * n))))
    end = max(start, min(n, int(round(end_ratio * n))))
    return "".join(chunks[start:end]) if lang.startswith(("zh", "ja", "ko")) else " ".join(chunks[start:end])


def text_context_by_audio_time(
    text: str,
    lang: str,
    start_s: float,
    end_s: float,
    total_s: float,
    context_s: float,
) -> tuple[str, float, float]:
    if total_s <= 0:
        return text, 0.0, 0.0
    context_start_s = max(0.0, start_s - context_s)
    context_end_s = min(total_s, end_s + context_s)
    return (
        text_span_by_ratio(text, lang, context_start_s / total_s, context_end_s / total_s),
        context_start_s,
        context_end_s,
    )


def load_candidate_texts(args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    if args.asr_jsonl:
        for row in read_jsonl(args.asr_jsonl):
            key = str(row.get("run_id") or row.get("id") or "")
            if key:
                out[key] = str(row.get("asr_text") or row.get("text") or "").strip()
    if args.asr_model:
        pipe = make_asr_pipe(args.asr_model, args.asr_device)
        for row in results:
            lang = str(row.get("target_lang") or row.get("target_language") or "")
            text = transcribe(pipe, row["generated_wav_path"], lang)
            out[str(row["run_id"])] = text
    return out


def load_window_texts(path: str | Path) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    if not path:
        return out
    for row in read_jsonl(path):
        if row.get("run_id") is None or row.get("window_index") is None:
            continue
        key = (str(row["run_id"]), int(row["window_index"]))
        out[key] = str(row.get("asr_text") or row.get("text") or "").strip()
    return out


def joined_window_text(
    run_id: str,
    window_texts: dict[tuple[str, int], str],
    target_lang: str,
) -> str:
    rows = [
        (window_index, text.strip())
        for (row_run_id, window_index), text in window_texts.items()
        if row_run_id == run_id and text.strip()
    ]
    if not rows:
        return ""
    rows.sort(key=lambda item: item[0])
    if str(target_lang or "").lower().startswith(("zh", "ja", "ko")):
        return "".join(text for _, text in rows)
    return " ".join(text for _, text in rows)


def load_target_mfa_texts(path: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    if not path:
        return out
    for row in read_jsonl(path):
        if row.get("run_id") is None or row.get("window_index") is None:
            continue
        key = (str(row["run_id"]), int(row["window_index"]))
        out[key] = row
    return out


def cumulative_at(chunks: list[dict[str, Any]], wall_s: float, *, inclusive: bool = True) -> float:
    value = 0.0
    for chunk in chunks:
        arrival_s = float(chunk.get("arrival_s", 0.0))
        hit = arrival_s <= wall_s if inclusive else arrival_s < wall_s
        if hit:
            value = max(value, float(chunk.get("cumulative_audio_duration_s", 0.0)))
    return value


def build_timeline(
    run: dict[str, Any],
    run_eval_dir: Path,
    *,
    candidate_text: str,
    window_texts: dict[tuple[str, int], str],
    target_mfa_texts: dict[tuple[str, int], dict[str, Any]],
    window_s: float,
    backlog_threshold_s: float,
    target_context_s: float,
    max_windows: int,
) -> list[dict[str, Any]]:
    source_duration_s = float(run.get("source_eval_duration_s") or audio_duration_s(run["source_eval_wav_path"]))
    speed = float(run["speed_factor"])
    source_wav = Path(str(run["source_eval_wav_path"]))
    source_stream_wav = Path(str(run.get("source_stream_wav_path") or run["source_eval_wav_path"]))
    generated_wav = Path(str(run["generated_wav_path"]))
    generated_duration_s = audio_duration_s(generated_wav)
    target_lang = str(run.get("target_lang") or run.get("target_language") or "")
    chunks = read_jsonl(run["audio_chunks_path"]) if Path(str(run["audio_chunks_path"])).exists() else []
    window_count = int((source_duration_s + window_s - 1e-9) // window_s)
    if max_windows > 0:
        window_count = min(window_count, max_windows)
    wav_dir = run_eval_dir / "windows"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(window_count):
        source_start = idx * window_s
        source_end = min(source_duration_s, source_start + window_s)
        wall_start = source_start / speed
        wall_end = source_end / speed
        emitted_start = cumulative_at(chunks, wall_start, inclusive=False)
        emitted_end = cumulative_at(chunks, wall_end, inclusive=True)
        queue_s = max(0.0, emitted_end - wall_end)
        backlog_s = max(0.0, wall_end - emitted_end)
        source_window = wav_dir / f"window_{idx:04d}_source.wav"
        source_stream_window = wav_dir / f"window_{idx:04d}_source_stream.wav"
        target_window = wav_dir / f"window_{idx:04d}_target.wav"
        slice_wav(source_wav, source_window, source_start, source_end)
        slice_wav(source_stream_wav, source_stream_window, wall_start, wall_end)
        slice_wav(generated_wav, target_window, emitted_start, emitted_end)
        context_text, context_start_s, context_end_s = text_context_by_audio_time(
            candidate_text,
            target_lang,
            emitted_start,
            emitted_end,
            generated_duration_s,
            target_context_s,
        )
        mfa_row = target_mfa_texts.get((str(run["run_id"]), idx), {})
        rows.append(
            {
                "run_id": run["run_id"],
                "window_index": idx,
                "source_window_start_s": round(source_start, 3),
                "source_window_end_s": round(source_end, 3),
                "speed_adjusted_input_start_s": round(wall_start, 3),
                "speed_adjusted_input_end_s": round(wall_end, 3),
                "target_audio_emitted_at_window_start_s": round(emitted_start, 6),
                "target_audio_emitted_until_boundary_s": round(emitted_end, 6),
                "target_window_duration_s": round(max(0.0, emitted_end - emitted_start), 6),
                "playback_queue_s": round(queue_s, 6),
                "translation_backlog_s": round(backlog_s, 6),
                "target_audio_deficit_s": round(backlog_s, 6),
                "backlog_violation": backlog_s > backlog_threshold_s,
                "source_window_audio_path": str(source_window),
                "source_stream_window_audio_path": str(source_stream_window),
                "target_window_audio_path": str(target_window),
                "source_window_audio_rel": rel(source_window, run_eval_dir),
                "source_stream_window_audio_rel": rel(source_stream_window, run_eval_dir),
                "target_window_audio_rel": rel(target_window, run_eval_dir),
                "target_asr_window_text": window_texts.get((str(run["run_id"]), idx), ""),
                "target_mfa_asr_text": str(mfa_row.get("target_mfa_asr_text") or ""),
                "target_mfa_start_s": mfa_row.get("target_mfa_start_s"),
                "target_mfa_end_s": mfa_row.get("target_mfa_end_s"),
                "target_mfa_unit_count": mfa_row.get("target_mfa_unit_count"),
                "target_full_asr_context_text": context_text,
                "target_full_asr_context_start_s": round(context_start_s, 3),
                "target_full_asr_context_end_s": round(context_end_s, 3),
            }
        )
    return rows


def judge_sentence(client: ChatClient, reference_sentence: str, candidate_text: str) -> dict[str, Any]:
    prompt = {
        "reference_sentence": reference_sentence,
        "candidate_transcript": candidate_text,
        "labels": ["covered", "partial", "missed"],
        "instruction": "Decide if the candidate speech transcript covers the meaning of the reference sentence. Return JSON only.",
    }
    messages = [
        {"role": "system", "content": "Return JSON only: {\"status\":\"covered|partial|missed\",\"reason\":\"...\"}"},
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ]
    data = extract_json_object(
        client.chat(messages, temperature=0.0, max_tokens=256, response_format={"type": "json_object"})
    )
    status = str(data.get("status") or "").strip().lower()
    if status not in {"covered", "partial", "missed"}:
        status = "partial"
    return {"llm_status": status, "llm_reason": str(data.get("reason") or "")[:500]}


def build_sentence_coverage(
    run: dict[str, Any],
    candidate_text: str,
    *,
    window_s: float,
    judge_client: ChatClient | None,
) -> list[dict[str, Any]]:
    rows = []
    eval_duration = float(run.get("source_eval_duration_s") or run.get("source_duration_s") or 0.0)
    for sent in run.get("target_sentences") or []:
        row = dict(sent)
        start_s = float(row.get("source_start_s") or 0.0)
        end_s = float(row.get("source_end_s") or start_s)
        if eval_duration > 0 and start_s >= eval_duration:
            continue
        target_sentence = str(row.get("target_sentence") or "")
        row.update(coverage_status(target_sentence, candidate_text))
        if judge_client is not None and target_sentence:
            try:
                row.update(judge_sentence(judge_client, target_sentence, candidate_text))
                row["status"] = row.get("llm_status") or row["status"]
            except Exception as exc:
                row["llm_error"] = f"{type(exc).__name__}: {exc}"
        row["run_id"] = run["run_id"]
        row["window_index"] = int(start_s // window_s)
        end_for_window = max(start_s, min(end_s, eval_duration) if eval_duration > 0 else end_s)
        last_window = int(max(start_s, end_for_window - 1e-6) // window_s)
        row["window_indexes"] = list(range(int(start_s // window_s), last_window + 1))
        rows.append(row)
    return rows


def reference_for_eval(run: dict[str, Any], sentence_rows: list[dict[str, Any]]) -> str:
    source_eval_duration = float(run.get("source_eval_duration_s") or 0.0)
    source_duration = float(run.get("source_duration_s") or 0.0)
    if source_eval_duration > 0 and source_duration > source_eval_duration + 1e-3:
        return " ".join(str(row.get("target_sentence") or "") for row in sentence_rows).strip()
    return str(run.get("target_reference_text") or "")


def final_audio_timing(run: dict[str, Any]) -> dict[str, float | None]:
    chunks_path = Path(str(run.get("audio_chunks_path") or ""))
    if not chunks_path.exists():
        return {"arrival_s": None, "playback_end_s": None}
    chunks = []
    for chunk in read_jsonl(chunks_path):
        try:
            arrival_s = float(chunk.get("arrival_s"))
        except (TypeError, ValueError):
            continue
        chunks.append((arrival_s, chunk))
    if not chunks:
        return {"arrival_s": None, "playback_end_s": None}

    playback_end_s = 0.0
    previous_cumulative_s = 0.0
    final_arrival_s = max(arrival_s for arrival_s, _chunk in chunks)
    for arrival_s, chunk in sorted(chunks, key=lambda item: item[0]):
        try:
            duration_s = float(chunk.get("audio_duration_s"))
        except (TypeError, ValueError):
            try:
                cumulative_s = float(chunk.get("cumulative_audio_duration_s"))
            except (TypeError, ValueError):
                cumulative_s = previous_cumulative_s
            duration_s = max(0.0, cumulative_s - previous_cumulative_s)
        if duration_s <= 0:
            continue
        playback_end_s = max(playback_end_s, arrival_s) + duration_s
        try:
            previous_cumulative_s = max(previous_cumulative_s, float(chunk.get("cumulative_audio_duration_s")))
        except (TypeError, ValueError):
            previous_cumulative_s += duration_s
    return {"arrival_s": final_arrival_s, "playback_end_s": playback_end_s if playback_end_s > 0 else None}


def run_metric_row(
    run: dict[str, Any],
    candidate_text: str,
    candidate_text_source: str,
    timeline: list[dict[str, Any]],
    sentence_rows: list[dict[str, Any]],
    run_page: Path,
    eval_dir: Path,
) -> dict[str, Any]:
    reference = reference_for_eval(run, sentence_rows)
    target_lang = str(run.get("target_lang") or run.get("target_language") or "")
    metrics = corpus_metrics(candidate_text, reference, target_lang)
    generated_duration = float(run.get("generated_duration_s") or audio_duration_s(run["generated_wav_path"]))
    source_stream_duration = float(run.get("source_stream_duration_s") or 0.0)
    final_timing = final_audio_timing(run)
    final_arrival = final_timing["arrival_s"]
    playback_end = final_timing["playback_end_s"]
    missed = [row for row in sentence_rows if row.get("status") == "missed"]
    partial = [row for row in sentence_rows if row.get("status") == "partial"]
    metrics.update(
        {
            "run_id": run["run_id"],
            "direction": run["direction"],
            "speed_factor": float(run["speed_factor"]),
            "source_eval_duration_s": float(run.get("source_eval_duration_s") or 0.0),
            "source_stream_duration_s": source_stream_duration,
            "generated_duration_s": round(generated_duration, 6),
            "full_s2s_rtf": round(generated_duration / source_stream_duration, 6)
            if source_stream_duration > 0
            else None,
            "end_lag_s": round(generated_duration - source_stream_duration, 6),
            "duration_end_lag_s": round(generated_duration - source_stream_duration, 6),
            "final_target_arrival_s": round(final_arrival, 6) if final_arrival is not None else None,
            "final_arrival_lag_s": round(final_arrival - source_stream_duration, 6)
            if final_arrival is not None and source_stream_duration > 0
            else None,
            "wall_clock_playback_end_s": round(playback_end, 6) if playback_end is not None else None,
            "wall_clock_end_delay_s": round(max(0.0, playback_end - source_stream_duration), 6)
            if playback_end is not None and source_stream_duration > 0
            else None,
            "max_backlog_s": round(max((row["translation_backlog_s"] for row in timeline), default=0.0), 6),
            "mean_backlog_s": round(
                sum(row["translation_backlog_s"] for row in timeline) / len(timeline), 6
            )
            if timeline
            else 0.0,
            "max_playback_queue_s": round(max((row["playback_queue_s"] for row in timeline), default=0.0), 6),
            "mean_playback_queue_s": round(
                sum(row["playback_queue_s"] for row in timeline) / len(timeline), 6
            )
            if timeline
            else 0.0,
            "max_deficit_s": round(max((row["target_audio_deficit_s"] for row in timeline), default=0.0), 6),
            "mean_deficit_s": round(
                sum(row["target_audio_deficit_s"] for row in timeline) / len(timeline), 6
            )
            if timeline
            else 0.0,
            "backlog_violation_rate": round(
                sum(1 for row in timeline if row.get("backlog_violation")) / len(timeline), 6
            )
            if timeline
            else 0.0,
            "deficit_violation_rate": round(
                sum(1 for row in timeline if row.get("target_audio_deficit_s", 0.0) > 0.5) / len(timeline),
                6,
            )
            if timeline
            else 0.0,
            "missed_sentence_rate": round(len(missed) / len(sentence_rows), 6) if sentence_rows else None,
            "partial_sentence_rate": round(len(partial) / len(sentence_rows), 6) if sentence_rows else None,
            "sentence_count": len(sentence_rows),
            "candidate_text": candidate_text,
            "candidate_text_source": candidate_text_source,
            "generated_wav_path": str(run["generated_wav_path"]),
            "source_audio_path": str(run.get("source_eval_wav_path") or ""),
            "source_stream_audio_path": str(run.get("source_stream_wav_path") or run.get("source_eval_wav_path") or ""),
            "run_page": str(run_page),
            "run_page_rel": rel(run_page, eval_dir),
        }
    )
    return metrics


def copy_full_audio(run: dict[str, Any], run_eval_dir: Path) -> None:
    source_copy = run_eval_dir / "source_eval.wav"
    source_stream_copy = run_eval_dir / "source_stream_24k.wav"
    generated_copy = run_eval_dir / "generated_target.wav"
    shutil.copy2(run["source_eval_wav_path"], source_copy)
    if run.get("source_stream_wav_path"):
        shutil.copy2(run["source_stream_wav_path"], source_stream_copy)
        run["source_stream_wav_path"] = str(source_stream_copy)
    shutil.copy2(run["generated_wav_path"], generated_copy)
    run["source_eval_wav_path"] = str(source_copy)
    run["generated_wav_path"] = str(generated_copy)


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.output_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    manifest_by_id = keyed(read_run_manifest(args.manifest), "run_id")
    results = load_results(Path(args.run_output_dir))
    candidate_by_id = load_candidate_texts(args, results)
    window_texts = load_window_texts(args.window_asr_jsonl)
    target_mfa_texts = load_target_mfa_texts(args.target_mfa_jsonl)
    judge_client = None
    if args.coverage_judge == "openai":
        import os

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY must be set for --coverage-judge openai")
        judge_client = ChatClient(
            base_url=args.openai_base_url.rstrip("/"),
            api_key=api_key,
            model=args.coverage_model,
            timeout_s=60.0,
        )

    metric_rows = []
    all_timeline = []
    all_sentences = []
    for result in results:
        run_id = str(result["run_id"])
        run = dict(manifest_by_id.get(run_id, {}))
        run.update(result)
        run_eval_dir = eval_dir / sanitize_id(run_id)
        run_eval_dir.mkdir(parents=True, exist_ok=True)
        copy_full_audio(run, run_eval_dir)
        target_lang = str(run.get("target_lang") or run.get("target_language") or "")
        window_candidate_text = joined_window_text(run_id, window_texts, target_lang)
        if window_candidate_text:
            candidate_text = window_candidate_text
            candidate_text_source = "window_asr_concat"
        elif candidate_by_id.get(run_id, "").strip():
            candidate_text = candidate_by_id[run_id].strip()
            candidate_text_source = "full_target_asr"
        elif args.asr_jsonl or args.asr_model:
            raise SystemExit(f"missing ASR transcript for {run_id}; refusing to fall back to output_transcript")
        else:
            candidate_text = str(run.get("output_transcript") or "")
            candidate_text_source = "output_transcript"
        timeline = build_timeline(
            run,
            run_eval_dir,
            candidate_text=candidate_text,
            window_texts=window_texts,
            target_mfa_texts=target_mfa_texts,
            window_s=args.window_s,
            backlog_threshold_s=args.backlog_threshold_s,
            target_context_s=args.target_context_s,
            max_windows=args.max_windows,
        )
        sentence_rows = build_sentence_coverage(
            run,
            candidate_text,
            window_s=args.window_s,
            judge_client=judge_client,
        )
        timeline_path = run_eval_dir / "timeline.jsonl"
        sentence_path = run_eval_dir / "sentence_coverage.jsonl"
        write_jsonl(timeline_path, timeline)
        write_jsonl(sentence_path, sentence_rows)
        run_page = run_eval_dir / "index.html"
        metrics = run_metric_row(
            run,
            candidate_text,
            candidate_text_source,
            timeline,
            sentence_rows,
            run_page,
            eval_dir,
        )
        (run_eval_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        write_run_dashboard(run_page, run, metrics, timeline, sentence_rows)
        metric_rows.append(metrics)
        all_timeline.extend(timeline)
        all_sentences.extend(sentence_rows)

    write_jsonl(eval_dir / "metrics.jsonl", metric_rows)
    write_jsonl(eval_dir / "timeline.jsonl", all_timeline)
    write_jsonl(eval_dir / "sentence_coverage.jsonl", all_sentences)
    summary = {
        "runs": len(metric_rows),
        "metrics": summarize_numeric(metric_rows),
        "timeline": summarize_numeric(all_timeline),
        "missed_sentences": sum(1 for row in all_sentences if row.get("status") == "missed"),
        "partial_sentences": sum(1 for row in all_sentences if row.get("status") == "partial"),
        "sentence_count": len(all_sentences),
    }
    (eval_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_combined_dashboard(eval_dir / "index.html", metric_rows)
    print(json.dumps({"runs": len(metric_rows), "output": str(eval_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
