#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
import wave
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s_omni.metrics import optional_sacrebleu, unit_count


RUN_ID = "en-zh_mono_asr_test__0__speed_1"


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 60s FLORAS EN->ZH compare dashboard including KIT Lecture Translator.",
    )
    parser.add_argument("--source-root", required=True, help="Local floras_live_pilot_refs output root.")
    parser.add_argument("--project-dir", required=True, help="projects/floras_live_s2s_benchmark directory.")
    parser.add_argument("--sacrebleu-path", default="", help="Optional directory containing sacrebleu.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reference_from_coverage(coverage_path: Path) -> str:
    rows = read_jsonl(coverage_path)
    return "".join(str(row.get("target_sentence") or "") for row in rows)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cer(reference: str, candidate: str) -> float | None:
    ref = list("".join((reference or "").split()))
    hyp = list("".join((candidate or "").split()))
    if not ref:
        return None if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, item in enumerate(ref, start=1):
        cur = [i]
        for j, cand in enumerate(hyp, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (item != cand)))
        prev = cur
    return int(prev[-1])


def load_reference(source_root: Path) -> str:
    coverage_path = source_root / "eval_real_60s" / RUN_ID / "sentence_coverage.jsonl"
    return reference_from_coverage(coverage_path)


def find_run_row(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    matches = [row for row in rows if str(row.get("run_id") or "") == RUN_ID]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {RUN_ID} row in {path}, found {len(matches)}")
    return matches[0]


def assert_same_reference(run_dir: Path, expected: str) -> str:
    coverage_path = run_dir / "sentence_coverage.jsonl"
    actual = reference_from_coverage(coverage_path)
    if actual != expected:
        raise ValueError(f"reference mismatch for {run_dir}: {sha256_text(actual)} != {sha256_text(expected)}")
    return str(coverage_path)


def assert_60s_window(row: dict[str, Any], eval_dir: Path) -> None:
    for key in ["source_eval_duration_s", "source_stream_duration_s"]:
        value = row.get(key)
        if value is None:
            raise ValueError(f"{eval_dir} missing {key}")
        if abs(float(value) - 60.0) > 0.01:
            raise ValueError(f"{eval_dir} {key} is {value}, expected 60.0")


def resolve_source_path(source_root: Path, value: Any) -> str:
    if not value:
        return ""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    candidates = [
        source_root / path,
        source_root.parents[1] / path,
        ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(path)


def first_existing(paths: list[Path]) -> str:
    for path in paths:
        if path.exists():
            return str(path)
    return ""


def wav_duration_s(path: str) -> float | None:
    if not path:
        return None
    wav_path = Path(path)
    if not wav_path.exists():
        return None
    with wave.open(str(wav_path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def compute_text_metrics(candidate: str, reference: str) -> dict[str, Any]:
    sacre = optional_sacrebleu([candidate], [reference], tokenizer="zh")
    default_sacre = optional_sacrebleu([candidate], [reference])
    out: dict[str, Any] = {
        "cer": cer(reference, candidate),
        "candidate_units": unit_count(candidate, "zh"),
        "reference_units": unit_count(reference, "zh"),
        "candidate_chars": len(candidate),
        "reference_chars": len(reference),
        "reference_sha256": sha256_text(reference),
        "bleu_tokenizer": "zh",
        "metric_inputs": "raw_hypothesis_and_reference_punctuation_preserved",
    }
    if sacre.get("available"):
        out["bleu"] = sacre.get("bleu")
        out["chrf"] = sacre.get("chrf")
    else:
        raise RuntimeError(f"sacreBLEU unavailable: {sacre.get('reason')}")
    if default_sacre.get("available"):
        out["bleu_default_tokenizer"] = default_sacre.get("bleu")
        out["bleu_zh_minus_default"] = round(float(out["bleu"]) - float(out["bleu_default_tokenizer"]), 6)
    return out


def metric_row(
    *,
    label: str,
    backend: str,
    chunk_ms: int | None,
    source: str,
    candidate: str,
    reference: str,
    comparison_scope: str,
    note: str,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = dict(base or {})
    row.update(compute_text_metrics(candidate, reference))
    row.update(
        {
            "eval_label": label,
            "compare_backend": backend,
            "compare_chunk_ms": chunk_ms,
            "run_id": RUN_ID,
            "direction": "en-zh",
            "speed_factor": 1.0,
            "candidate_text_source": source,
            "comparison_scope": comparison_scope,
            "note": note,
            "candidate_text": candidate,
            "hypothesis_text": candidate,
            "reference_text": reference,
        }
    )
    return row


def metrics_file_row(
    *,
    source_root: Path,
    rel_dir: str,
    label: str,
    backend: str,
    chunk_ms: int,
    reference: str,
    note: str,
) -> dict[str, Any]:
    eval_dir = source_root / rel_dir
    metrics_path = eval_dir / "metrics.jsonl"
    row = find_run_row(read_jsonl(metrics_path), metrics_path)
    assert_60s_window(row, eval_dir)
    run_dir = eval_dir / RUN_ID
    coverage_path = assert_same_reference(run_dir, reference)
    candidate = str(row.get("candidate_text") or "")
    base = {
        key: row.get(key)
        for key in [
            "source_eval_duration_s",
            "source_stream_duration_s",
            "generated_duration_s",
            "full_s2s_rtf",
            "end_lag_s",
            "duration_end_lag_s",
            "max_backlog_s",
            "mean_backlog_s",
            "wall_clock_end_delay_s",
            "max_playback_queue_s",
        ]
        if key in row
    }
    base["source_audio_path"] = first_existing(
        [
            run_dir / "source_eval.wav",
            run_dir / "source_stream_24k.wav",
            run_dir / "source_stream_16k.wav",
        ]
    )
    base["source_stream_audio_path"] = first_existing(
        [
            run_dir / "source_stream_24k.wav",
            run_dir / "source_stream_16k.wav",
            run_dir / "source_eval.wav",
        ]
    )
    base["generated_audio_path"] = resolve_source_path(source_root, row.get("generated_wav_path"))
    base["run_page_path"] = resolve_source_path(source_root, row.get("run_page") or run_dir / "index.html")
    base["reference_source_path"] = coverage_path
    base["reference_validation"] = "matched_run_sentence_coverage_jsonl"
    return metric_row(
        label=label,
        backend=backend,
        chunk_ms=chunk_ms,
        source="target_speech_asr_gpt4o_mini_transcribe",
        candidate=candidate,
        reference=reference,
        comparison_scope="exact_60s_source_clip",
        note=note,
        base=base,
    )


def load_kit_rows(source_root: Path, reference: str) -> list[dict[str, Any]]:
    diagnosis = read_json(source_root / "kit_upload_smoke" / "kit_extractor_diagnosis.json")
    fast_run = read_json(source_root / "kit_upload_smoke" / "kit_live_enzh_run.json")
    realtime_run = read_json(source_root / "kit_upload_smoke" / "kit_live_enzh_realtime_run.json")
    source_audio = first_existing(
        [
            source_root / "kit_upload_smoke" / "source_60s.wav",
            source_root / "eval_real_60s" / RUN_ID / "source_stream_24k.wav",
            source_root / "eval_real_60s" / RUN_ID / "source_eval.wav",
        ]
    )
    common = {
        "source_eval_duration_s": 60.0,
        "source_stream_duration_s": 60.0,
        "source_audio_path": source_audio,
        "source_stream_audio_path": source_audio,
    }
    rows = []
    for mode, run in [("fast", fast_run), ("realtime", realtime_run)]:
        tts_text = str(diagnosis["runs"][mode]["tts_text"])
        session_url = str(run.get("sessionUrl") or "")
        base = dict(common)
        base["valid_for_main_s2s_target_audio"] = False
        base["run_page_path"] = session_url
        base["kit_session_url"] = session_url
        if mode == "fast":
            base["kit_feed_mode"] = "accelerated_1s_audio_every_120ms"
            note = "Debug only: KIT web-event TTS text from accelerated upload, not target-speech ASR."
        else:
            base["kit_feed_mode"] = "realtime_1s_audio_every_1s"
            note = "Debug only: KIT web-event TTS text from realtime-paced upload, not target-speech ASR."
        rows.append(
            metric_row(
                label=f"kit_{mode}_60s_tts_text",
                backend="kit_lecture_translator",
                chunk_ms=None,
                source="kit_websocket_tts_text",
                candidate=tts_text,
                reference=reference,
                comparison_scope="debug_text_only_exact_60s_source_clip",
                note=note,
                base=base,
            )
        )

    target_asr_metrics = source_root / "kit_config_smoke_60s_chunk1920" / "target_tts_asr_metrics.jsonl"
    if target_asr_metrics.exists():
        label_map = {
            "online_low_latency_no_post": "kit_online_low_latency_target_asr",
            "online_high_quality_no_post": "kit_online_high_quality_target_asr",
            "mixed_high_quality_no_post": "kit_mixed_high_quality_target_asr",
        }
        for row in read_jsonl(target_asr_metrics):
            config = str(row.get("config") or "")
            if config not in label_map:
                continue
            audio_path = str(row.get("target_audio_path") or "")
            duration = wav_duration_s(audio_path)
            base = dict(common)
            base.update(
                {
                    "generated_audio_path": audio_path,
                    "reference_validation": "matched_60s_reference_text",
                    "kit_config": config,
                    "valid_for_main_s2s_target_audio": not config.startswith("mixed_"),
                }
            )
            if duration is not None:
                base["generated_duration_s"] = round(duration, 6)
                base["end_lag_s"] = round(duration - 60.0, 6)
            if config.startswith("mixed_"):
                scope = "debug_revision_mode_exact_60s_source_clip"
                note = (
                    "Debug only: KIT format=mixed may revise already emitted sentences; "
                    "do not use as the main S2S target-audio metric."
                )
            else:
                scope = "exact_60s_source_clip"
                note = "KIT target speech retrieved from linked tts:0 data and scored through gpt-4o-mini-transcribe."
            rows.append(
                metric_row(
                    label=label_map[config],
                    backend="kit_lecture_translator",
                    chunk_ms=1920,
                    source="target_speech_asr_gpt4o_mini_transcribe",
                    candidate=str(row.get("hypothesis_text") or row.get("asr_text") or ""),
                    reference=reference,
                    comparison_scope=scope,
                    note=note,
                    base=base,
                )
            )

    en_only_metrics = (
        source_root
        / "kit_profile_minimal_60s_chunk1920"
        / "online_high_quality_en_only_no_summarization"
        / "compare_online_en_only_vs_prior_metrics.json"
    )
    if en_only_metrics.exists():
        row = read_json(en_only_metrics)["new_row"]
        audio_path = str(row.get("target_audio_path") or "")
        base = dict(common)
        base.update(
            {
                "generated_audio_path": audio_path,
                "generated_duration_s": row.get("target_duration_s"),
                "end_lag_s": round(float(row.get("target_duration_s") or 60.0) - 60.0, 6),
                "reference_validation": "matched_60s_reference_text",
                "kit_config": row.get("config"),
                "kit_language": "en",
                "kit_mt_language": "zh",
                "kit_audio_language": "zh",
                "kit_tts_quality_mode": "high_quality",
                "valid_for_main_s2s_target_audio": True,
                "kit_tts_audio_chunks": row.get("tts_audio_chunks"),
            }
        )
        rows.append(
            metric_row(
                label="kit_online_high_quality_enonly_target_asr",
                backend="kit_lecture_translator",
                chunk_ms=1920,
                source="target_speech_asr_gpt4o_mini_transcribe",
                candidate=str(row.get("hypothesis_text") or row.get("asr_text") or ""),
                reference=reference,
                comparison_scope="exact_60s_source_clip",
                note=(
                    "KIT explicit minimal online run: language=en, mtLanguage=zh, "
                    "audioLanguage=zh, ttsQualityMode=high_quality."
                ),
                base=base,
            )
        )
    return rows


def cjk_prefix_preserve_punctuation(text: str, units: int) -> str:
    out: list[str] = []
    reached = False
    for ch in text:
        if reached and "\u4e00" <= ch <= "\u9fff":
            break
        out.append(ch)
        if "\u4e00" <= ch <= "\u9fff":
            units -= 1
            if units <= 0:
                reached = True
    return "".join(out)


def seed_proxy_row(
    *,
    project_dir: Path,
    seed_rel_dir: str,
    label: str,
    chunk_ms: int,
    reference: str,
) -> dict[str, Any]:
    metrics_path = project_dir / "artifacts" / "eval_runs" / seed_rel_dir / "metrics.jsonl"
    row = find_run_row(read_jsonl(metrics_path), metrics_path)
    candidate_full = str(row.get("candidate_text") or "")
    candidate = cjk_prefix_preserve_punctuation(candidate_full, unit_count(reference, "zh"))
    base = {
        key: row.get(key)
        for key in [
            "source_eval_duration_s",
            "source_stream_duration_s",
            "generated_duration_s",
            "full_s2s_rtf",
            "end_lag_s",
            "duration_end_lag_s",
            "max_backlog_s",
            "mean_backlog_s",
            "wall_clock_end_delay_s",
            "max_playback_queue_s",
        ]
        if key in row
    }
    base["seed_full_bleu"] = row.get("bleu")
    base["seed_full_chrf"] = row.get("chrf")
    base["seed_full_cer"] = row.get("cer")
    base["run_page_path"] = str(project_dir / "artifacts" / "eval_runs" / seed_rel_dir / RUN_ID / "index.html")
    base["reference_validation"] = "scored_against_eval_real_60s_reference_proxy_only"
    return metric_row(
        label=label,
        backend="seed_ast",
        chunk_ms=chunk_ms,
        source="full_target_speech_asr_prefix_proxy_gpt4o_mini_transcribe",
        candidate=candidate,
        reference=reference,
        comparison_scope="proxy_prefix_from_full_1072s_seed_run",
        note="Proxy only: Seed was not rerun on the 60s clip; prefix was cut by CJK reference unit count with punctuation kept.",
        base=base,
    )


def href(value: Any, base: Path) -> str:
    if not value:
        return ""
    raw = str(value)
    if raw.startswith(("http://", "https://")):
        return raw
    path = Path(raw)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            return path.resolve().as_uri()
    return raw


def audio(value: Any, base: Path) -> str:
    if not value:
        return ""
    raw = str(value)
    if raw.startswith(("http://", "https://")):
        return ""
    path = Path(raw)
    if not path.exists():
        return ""
    return f'<audio controls preload="metadata" src="{esc(href(path, base))}"></audio>'


def link(value: Any, base: Path, text: str) -> str:
    if not value:
        return ""
    raw = str(value)
    if raw.startswith(("http://", "https://")):
        return f'<a href="{esc(raw)}">{esc(text)}</a>'
    path = Path(raw)
    if path.exists():
        return f'<a href="{esc(href(path, base))}">{esc(text)}</a>'
    return ""


def num(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    backend_order = {
        "chatgpt": 0,
        "gemini": 1,
        "kit_lecture_translator": 2,
        "seed_ast": 3,
    }
    return (
        int(row.get("compare_chunk_ms") or 999999),
        backend_order.get(str(row.get("compare_backend")), 99),
        str(row.get("eval_label")),
    )


def render_table(rows: list[dict[str, Any]], out_dir: Path) -> str:
    table_rows = []
    for row in sorted(rows, key=sort_key):
        backend = row.get("compare_backend")
        chunk = row.get("compare_chunk_ms")
        chunk_label = "" if chunk is None else f"{float(chunk) / 1000.0:.2f}s"
        label = str(row.get("eval_label") or backend)
        table_rows.append(
            "<tr>"
            f"<td>{esc(label)}</td>"
            f"<td>{esc(backend)}</td>"
            f"<td>{esc(chunk_label)}</td>"
            f"<td>{esc(row.get('comparison_scope'))}</td>"
            f"<td>{num(row.get('bleu_default_tokenizer'))}</td>"
            f"<td>{num(row.get('bleu'))}</td>"
            f"<td>{num(row.get('chrf'))}</td>"
            f"<td>{num(row.get('cer'), 3)}</td>"
            f"<td>{num(row.get('source_stream_duration_s'))}</td>"
            f"<td>{num(row.get('generated_duration_s'))}</td>"
            f"<td>{num(row.get('end_lag_s'))}</td>"
            f"<td>{num(row.get('wall_clock_end_delay_s'))}</td>"
            f"<td>{num(row.get('max_backlog_s'))}</td>"
            f"<td>{esc(row.get('candidate_text_source'))}</td>"
            f"<td>{link(row.get('run_page_path'), out_dir, 'open')}</td>"
            f"<td>{audio(row.get('source_stream_audio_path') or row.get('source_audio_path'), out_dir)}</td>"
            f"<td>{audio(row.get('generated_audio_path'), out_dir)}</td>"
            f"<td>{esc(row.get('note'))}</td>"
            "</tr>"
        )
    return "\n".join(table_rows)


def render_details(rows: list[dict[str, Any]]) -> str:
    detail_rows = []
    for row in sorted(rows, key=sort_key):
        label = str(row.get("eval_label") or row.get("compare_backend"))
        detail_rows.append(
            f"<details><summary>{esc(label)} · BLEU {num(row.get('bleu'))} · chrF {num(row.get('chrf'))}</summary>"
            f"<div class=\"textgrid\"><div><h3>Hypothesis</h3><pre>{esc(row.get('hypothesis_text'))}</pre></div>"
            f"<div><h3>Reference</h3><pre>{esc(row.get('reference_text'))}</pre></div></div>"
            "</details>"
        )
    return "\n".join(detail_rows)


def render_html(rows: list[dict[str, Any]], out_dir: Path) -> str:
    measured_rows = [
        row
        for row in rows
        if not str(row.get("comparison_scope") or "").startswith(("proxy_", "debug_"))
        and row.get("valid_for_main_s2s_target_audio") is not False
    ]
    debug_rows = [
        row
        for row in rows
        if str(row.get("comparison_scope") or "").startswith("debug_")
        or row.get("valid_for_main_s2s_target_audio") is False
    ]
    proxy_rows = [row for row in rows if str(row.get("comparison_scope") or "").startswith("proxy_")]
    debug_section = ""
    if debug_rows:
        debug_section = f"""
<h2>Debug / Non-Main Rows</h2>
<p class="meta">
Rows here are not main emitted target-speech metrics. KIT `format=mixed` may revise already emitted sentences;
KIT web-event TTS text is not target-speech ASR.
</p>
<table>
<thead><tr>
<th>label</th><th>backend</th><th>chunk</th><th>scope</th><th>BLEU default</th><th>BLEU zh</th><th>chrF</th><th>CER</th>
<th>source s</th><th>target s</th><th>duration lag</th><th>wall delay</th><th>max backlog</th>
<th>text source</th><th>detail</th><th>source audio</th><th>target audio</th><th>note</th>
</tr></thead>
<tbody>
{render_table(debug_rows, out_dir)}
</tbody>
</table>
"""
    return f"""<!doctype html>
<meta charset="utf-8">
<title>FLORAS EN-ZH 60s Live S2S Compare</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;color:#202428;background:#fbfbfc}}
h1{{font-size:22px;margin:0 0 8px}}.meta{{color:#667085;margin:0 0 18px;line-height:1.45}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{border-top:1px solid #d6d9de;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{background:#f2f4f7;font-weight:600}}audio{{width:210px}}details{{margin:14px 0;padding:12px;background:white;border:1px solid #d6d9de;border-radius:6px}}
summary{{cursor:pointer;font-weight:600}}.textgrid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}}pre{{white-space:pre-wrap;word-break:break-word;font-size:13px;line-height:1.45;background:#f8fafc;padding:10px;border-radius:6px}}
h3{{font-size:13px;margin:0 0 6px}}@media(max-width:900px){{.textgrid{{grid-template-columns:1fr}}table{{display:block;overflow-x:auto}}}}
</style>
<h1>FLORAS EN-ZH 60s Live S2S Compare</h1>
<p class="meta">
{len(rows)} rows over the same first 60s FLORAS EN-&gt;ZH source clip. BLEU uses sacreBLEU tokenize=zh.
Hypothesis/reference strings are stored and displayed with punctuation preserved. CER ignores whitespace only and keeps punctuation.
Main KIT rows use retrieved target speech scored through gpt-4o-mini-transcribe. KIT mixed-mode/text-only rows are separated as debug rows.
Seed rows are marked as proxy prefixes from full 1072s runs.
</p>
<h2>Main Exact 60s Measurements</h2>
<table>
<thead><tr>
<th>label</th><th>backend</th><th>chunk</th><th>scope</th><th>BLEU default</th><th>BLEU zh</th><th>chrF</th><th>CER</th>
<th>source s</th><th>target s</th><th>duration lag</th><th>wall delay</th><th>max backlog</th>
<th>text source</th><th>detail</th><th>source audio</th><th>target audio</th><th>note</th>
</tr></thead>
<tbody>
{render_table(measured_rows, out_dir)}
</tbody>
</table>
{debug_section}
<h2>Seed Proxy Rows</h2>
<p class="meta">These rows are not exact 60s reruns; they are full-run target-speech ASR prefixes cut by the 60s reference CJK unit count.</p>
<table>
<thead><tr>
<th>label</th><th>backend</th><th>chunk</th><th>scope</th><th>BLEU default</th><th>BLEU zh</th><th>chrF</th><th>CER</th>
<th>source s</th><th>target s</th><th>duration lag</th><th>wall delay</th><th>max backlog</th>
<th>text source</th><th>detail</th><th>source audio</th><th>target audio</th><th>note</th>
</tr></thead>
<tbody>
{render_table(proxy_rows, out_dir)}
</tbody>
</table>
<h2>Hypothesis / Reference</h2>
{render_details(rows)}
"""


def main() -> None:
    args = parse_args()
    if args.sacrebleu_path:
        sys.path.insert(0, str(Path(args.sacrebleu_path).expanduser()))
    source_root = Path(args.source_root).expanduser().resolve()
    project_dir = Path(args.project_dir).expanduser().resolve()
    out_dir = project_dir / "artifacts" / "compare_gpt_gemini_seed_kit_enzh_60s"
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = load_reference(source_root)
    rows = [
        metrics_file_row(
            source_root=source_root,
            rel_dir="eval_real_60s",
            label="chatgpt_default_60s_asr",
            backend="chatgpt",
            chunk_ms=960,
            reference=reference,
            note="OpenAI Realtime 60s smoke result; BLEU recomputed with zh tokenizer over raw text.",
        ),
        metrics_file_row(
            source_root=source_root,
            rel_dir="openai_eval_enzh_60s_chunk1920_asr",
            label="chatgpt_chunk1920_60s_asr",
            backend="chatgpt",
            chunk_ms=1920,
            reference=reference,
            note="OpenAI Realtime 60s chunk1920 result; BLEU recomputed with zh tokenizer over raw text.",
        ),
        metrics_file_row(
            source_root=source_root,
            rel_dir="gemini_eval_enzh_60s_trim_asr",
            label="gemini_default_60s_asr",
            backend="gemini",
            chunk_ms=960,
            reference=reference,
            note="Gemini Live 60s trimmed result; BLEU recomputed with zh tokenizer over raw text.",
        ),
        metrics_file_row(
            source_root=source_root,
            rel_dir="gemini_eval_enzh_60s_chunk1920_trim_asr",
            label="gemini_chunk1920_60s_asr",
            backend="gemini",
            chunk_ms=1920,
            reference=reference,
            note="Gemini Live 60s chunk1920 trimmed result; BLEU recomputed with zh tokenizer over raw text.",
        ),
    ]
    rows.extend(load_kit_rows(source_root, reference))
    rows.extend(
        [
            seed_proxy_row(
                project_dir=project_dir,
                seed_rel_dir="seed_ast_chunk960_gpt4o_mini_asr",
                label="seed_ast_chunk960_prefix_proxy",
                chunk_ms=960,
                reference=reference,
            ),
            seed_proxy_row(
                project_dir=project_dir,
                seed_rel_dir="seed_ast_chunk1920_gpt4o_mini_asr",
                label="seed_ast_chunk1920_prefix_proxy",
                chunk_ms=1920,
                reference=reference,
            ),
        ]
    )
    sorted_rows = sorted(rows, key=sort_key)
    (out_dir / "compare_metrics.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in sorted_rows) + "\n",
        encoding="utf-8",
    )
    summary = {
        "run_id": RUN_ID,
        "reference_units": unit_count(reference, "zh"),
        "reference_chars": len(reference),
        "reference_sha256": sha256_text(reference),
        "bleu_tokenizer": "zh",
        "metric_inputs": "punctuation preserved in hypothesis/reference",
        "rows": [
            {
                "label": row["eval_label"],
                "backend": row["compare_backend"],
                "chunk_ms": row["compare_chunk_ms"],
                "scope": row["comparison_scope"],
                "bleu_default_tokenizer": row.get("bleu_default_tokenizer"),
                "bleu": row["bleu"],
                "chrf": row["chrf"],
                "cer": row["cer"],
                "valid_for_main_s2s_target_audio": row.get("valid_for_main_s2s_target_audio", True),
            }
            for row in sorted_rows
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "index.html").write_text(render_html(sorted_rows, out_dir), encoding="utf-8")
    print(json.dumps({"rows": len(sorted_rows), "output_dir": str(out_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
