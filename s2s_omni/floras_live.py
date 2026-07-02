from __future__ import annotations

import base64
import html
import json
import math
import os
import re
import shutil
import subprocess
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io import read_jsonl, write_jsonl
from .metrics import bag_f1, optional_sacrebleu, tokenize, unit_count
from .rasst import audio_duration_s, atempo_filters, load_mono_audio, write_mono_wav


FLORES_REPO = "espnet/floras"
OPENAI_TRANSLATION_MODEL = "gpt-realtime-translate"
OPENAI_TRANSLATION_WS = "wss://api.openai.com/v1/realtime/translations"
GEMINI_TRANSLATION_MODEL = "gemini-3.5-live-translate-preview"
GEMINI_TRANSLATION_WS = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
GEMINI_INPUT_SAMPLE_RATE = 16000
SOURCE_WINDOW_S = 10.0
REALTIME_SAMPLE_RATE = 24000
REALTIME_CHUNK_MS = 960


@dataclass(frozen=True)
class FlorasSelection:
    direction: str
    source_lang: str
    target_lang: str
    id: str
    score: float
    source_duration_s: float
    source_wav_path: str
    source_transcript: str
    target_reference_text: str
    summary: str
    source_shard: str
    reference_kind: str
    target_sentences: list[dict[str, Any]]


def sanitize_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("_") or "sample"


def parse_float(value: Any, default: float = math.inf) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_speeds(raw: str) -> list[float]:
    speeds = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not speeds:
        raise ValueError("speed list cannot be empty")
    for speed in speeds:
        if speed <= 0:
            raise ValueError(f"invalid speed factor: {speed}")
    return speeds


def split_sentences(text: str, lang: str) -> list[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    lang = lang.lower()
    if lang.startswith("zh"):
        pieces = re.split(r"(?<=[。！？!?])\s*", text)
    elif lang.startswith("ja"):
        pieces = re.split(r"(?<=[。！？!?])\s*", text)
    else:
        pieces = re.split(r"(?<=[.!?])\s+", text)
    out = [piece.strip() for piece in pieces if piece.strip()]
    if len(out) <= 1 and len(text) > 800:
        out = chunk_text_by_units(text, lang, max_units=80)
    return out


def chunk_text_by_units(text: str, lang: str, max_units: int) -> list[str]:
    words = text.split()
    if not lang.lower().startswith(("zh", "ja")) and len(words) > 1:
        chunks = []
        for start in range(0, len(words), max_units):
            chunks.append(" ".join(words[start : start + max_units]).strip())
        return [chunk for chunk in chunks if chunk]
    chunks = []
    current = []
    current_units = 0
    for ch in text:
        current.append(ch)
        current_units += 1
        if current_units >= max_units and ch.strip():
            chunks.append("".join(current).strip())
            current = []
            current_units = 0
    if current:
        chunks.append("".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def proportional_sentence_rows(
    sentences: list[str],
    source_duration_s: float,
    target_lang: str,
    translated_sentences: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not sentences:
        return rows
    total_units = sum(max(1, unit_count(sentence, target_lang)) for sentence in sentences)
    cursor = 0.0
    for index, sentence in enumerate(sentences):
        units = max(1, unit_count(sentence, target_lang))
        start_s = cursor / max(1, total_units) * source_duration_s
        cursor += units
        end_s = cursor / max(1, total_units) * source_duration_s
        target_sentence = (
            translated_sentences[index].strip()
            if translated_sentences and index < len(translated_sentences)
            else sentence
        )
        rows.append(
            {
                "sentence_index": index,
                "source_start_s": round(start_s, 3),
                "source_end_s": round(end_s, 3),
                "source_sentence": sentence,
                "target_sentence": target_sentence,
                "timing_method": "proportional_text_units",
            }
        )
    return rows


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")


def ffmpeg_convert(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sample_rate: int = REALTIME_SAMPLE_RATE,
    speed: float = 1.0,
    duration_s: float | None = None,
    start_s: float | None = None,
    overwrite: bool = True,
) -> None:
    ensure_ffmpeg()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start_s is not None and start_s > 0:
        cmd.extend(["-ss", f"{start_s:.6f}"])
    cmd.extend(["-y" if overwrite else "-n", "-i", str(input_path)])
    if duration_s is not None and duration_s > 0:
        cmd.extend(["-t", f"{duration_s:.6f}"])
    filters = []
    if abs(speed - 1.0) > 1e-6:
        filters.append(atempo_filters(speed))
    if filters:
        cmd.extend(["-filter:a", ",".join(filters)])
    cmd.extend(["-ac", "1", "-ar", str(sample_rate), str(output_path)])
    subprocess.run(cmd, check=True)


def slice_wav(input_path: str | Path, output_path: str | Path, start_s: float, end_s: float) -> None:
    if end_s <= start_s:
        write_mono_wav(output_path, np.zeros(1, dtype=np.float32), REALTIME_SAMPLE_RATE)
        return
    wav, sr = load_mono_audio(input_path)
    start = max(0, min(len(wav), int(round(start_s * sr))))
    end = max(start, min(len(wav), int(round(end_s * sr))))
    write_mono_wav(output_path, wav[start:end], sr)


def wav_info(path: str | Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        sr = handle.getframerate()
        channels = handle.getnchannels()
    return {
        "sample_rate": sr,
        "channels": channels,
        "num_frames": frames,
        "duration_s": round(frames / sr if sr else 0.0, 6),
    }


def wav_to_pcm16(path: str | Path, sample_rate: int = REALTIME_SAMPLE_RATE) -> bytes:
    wav, sr = load_mono_audio(path, sample_rate)
    pcm = np.clip(wav, -1.0, 1.0)
    return (pcm * 32767.0).astype("<i2").tobytes()


def pcm16_to_wav(path: str | Path, payload: bytes, sample_rate: int = REALTIME_SAMPLE_RATE) -> float:
    if payload:
        audio = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32767.0
    else:
        audio = np.zeros(1, dtype=np.float32)
    write_mono_wav(path, audio, sample_rate)
    return float(audio.shape[0]) / float(sample_rate)


def pcm16_duration_s(payload: bytes, sample_rate: int = REALTIME_SAMPLE_RATE) -> float:
    return len(payload) / 2.0 / float(sample_rate)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


def hf_resolve_url(repo_id: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path}?download=true"


def download_file(url: str, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with urllib.request.urlopen(url) as resp, tmp_path.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp_path.replace(output_path)
    return output_path


def load_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required to read FLORAS parquet shards") from exc
    return pq.read_table(path).to_pylist()


def export_floras_audio(row: dict[str, Any], output_path: str | Path) -> float:
    audio = row.get("audio") or {}
    payload = audio.get("bytes")
    if payload is None:
        raise ValueError(f"row {row.get('id')} has no embedded audio bytes")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    return audio_duration_s(output_path)


def source_text_without_lang_tag(text: str) -> str:
    return re.sub(r"^\[[a-zA-Z_-]+\]\s*", "", text or "").strip()


def wer(reference: str, candidate: str) -> float | None:
    ref = tokenize(reference)
    hyp = tokenize(candidate)
    if not ref:
        return None if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


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


def coverage_status(reference_sentence: str, candidate_text: str) -> dict[str, Any]:
    score = bag_f1(reference_sentence, candidate_text or "") or 0.0
    ref_tokens = tokenize(reference_sentence)
    cand_tokens = set(tokenize(candidate_text or ""))
    recall = 0.0
    if ref_tokens:
        recall = sum(1 for token in ref_tokens if token in cand_tokens) / len(ref_tokens)
    if recall >= 0.75 or score >= 0.5:
        status = "covered"
    elif recall >= 0.35 or score >= 0.2:
        status = "partial"
    else:
        status = "missed"
    return {
        "status": status,
        "coverage_method": "heuristic_lexical",
        "heuristic_bag_f1": round(float(score), 6),
        "heuristic_recall": round(float(recall), 6),
    }


def summarize_numeric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(rows)}
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    for key in keys:
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if vals:
            vals_sorted = sorted(vals)
            out[key] = {
                "mean": round(float(sum(vals) / len(vals)), 6),
                "min": round(vals_sorted[0], 6),
                "p50": round(percentile(vals_sorted, 0.5), 6),
                "p90": round(percentile(vals_sorted, 0.9), 6),
                "max": round(vals_sorted[-1], 6),
            }
    return out


def percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = min(len(values) - 1, lo + 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def corpus_metrics(candidate: str, reference: str, target_lang: str) -> dict[str, Any]:
    target_lang_norm = (target_lang or "").lower()
    bleu_tokenizer = None
    if target_lang_norm.startswith("zh"):
        bleu_tokenizer = "zh"
    elif target_lang_norm.startswith(("ja", "ko")):
        bleu_tokenizer = "char"
    out: dict[str, Any] = {
        "wer": wer(reference, candidate),
        "cer": cer(reference, candidate),
        "candidate_units": unit_count(candidate, target_lang),
        "reference_units": unit_count(reference, target_lang),
    }
    sacre = optional_sacrebleu([candidate], [reference], tokenizer=bleu_tokenizer)
    if sacre.get("available"):
        out["bleu"] = sacre.get("bleu")
        out["chrf"] = sacre.get("chrf")
        if sacre.get("bleu_tokenizer"):
            out["bleu_tokenizer"] = sacre.get("bleu_tokenizer")
    else:
        out["bleu"] = None
        out["chrf"] = None
        out["sacrebleu_unavailable_reason"] = sacre.get("reason")
    for key, value in list(out.items()):
        if isinstance(value, float):
            out[key] = round(value, 6)
    return out


def read_run_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for row in rows:
        for key in ["run_id", "direction", "speed_factor", "source_wav_path"]:
            if key not in row:
                raise ValueError(f"{path}: missing required field {key}")
    return rows


def rel(path: str | Path, base: str | Path) -> str:
    path = Path(path)
    base = Path(base)
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def env_api_key(name: str = "OPENAI_API_KEY") -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def target_language_for_direction(direction: str) -> str:
    if direction == "zh-en":
        return "en"
    if direction == "en-zh":
        return "zh"
    raise ValueError(f"unsupported direction: {direction}")


def source_language_for_direction(direction: str) -> str:
    if direction == "zh-en":
        return "zh"
    if direction == "en-zh":
        return "en"
    raise ValueError(f"unsupported direction: {direction}")


def write_run_dashboard(
    output_path: str | Path,
    run: dict[str, Any],
    metrics: dict[str, Any],
    timeline: list[dict[str, Any]],
    sentence_rows: list[dict[str, Any]],
) -> None:
    output_path = Path(output_path)
    run_dir = output_path.parent
    sentence_by_window: dict[int, list[dict[str, Any]]] = {}
    for row in sentence_rows:
        sentence_by_window.setdefault(int(row.get("window_index", 0)), []).append(row)
    sections = []
    for window in timeline:
        idx = int(window["window_index"])
        sent_html = "\n".join(
            f"<li class='{esc(sent.get('status'))}'><b>{esc(sent.get('status'))}</b> "
            f"<span class='score'>"
            f"{esc(sent.get('coverage_method'))} · recall {esc(sent.get('heuristic_recall'))} · "
            f"F1 {esc(sent.get('heuristic_bag_f1'))}"
            f"</span>"
            f"<span class='source-sentence'>{esc(sent.get('source_sentence'))}</span>"
            f"<span>{esc(sent.get('target_sentence'))}</span></li>"
            for sent in sentence_by_window.get(idx, [])
        )
        sections.append(
            f"""
<section>
  <h2>Window {idx:03d}: {window['source_window_start_s']:.1f}s-{window['source_window_end_s']:.1f}s</h2>
  <div class="meta">translation backlog {window['translation_backlog_s']:.3f}s · playback queue {window['playback_queue_s']:.3f}s · target audio {window['target_audio_emitted_at_window_start_s']:.3f}s-{window['target_audio_emitted_until_boundary_s']:.3f}s · violation {esc(window['backlog_violation'])}</div>
  <div class="actions">
    <button type="button" data-seek-audio="full-source-stream" data-seek-s="{window['speed_adjusted_input_start_s']:.3f}">jump streamed source</button>
    <button type="button" data-seek-audio="full-target-audio" data-seek-s="{window['target_audio_emitted_at_window_start_s']:.3f}">jump full target</button>
  </div>
  <div class="grid">
    <div><div class="label">source original window</div><audio controls preload="metadata" src="{esc(window.get('source_window_audio_rel'))}"></audio></div>
    <div><div class="label">source streamed window</div><audio controls preload="metadata" src="{esc(window.get('source_stream_window_audio_rel'))}"></audio></div>
    <div><div class="label">target generated by arrival</div><audio controls preload="metadata" src="{esc(window.get('target_window_audio_rel'))}"></audio></div>
  </div>
  <div class="label">target MFA-aligned ASR ({esc(window.get('target_mfa_start_s'))}s-{esc(window.get('target_mfa_end_s'))}s)</div><p>{esc(window.get('target_mfa_asr_text') or 'No MFA-aligned ASR available.')}</p>
  <div class="label">target arrival-window ASR</div><p>{esc(window.get('target_asr_window_text') or 'No per-window ASR available.')}</p>
  <div class="label">rough full-target ASR context ({esc(window.get('target_full_asr_context_start_s'))}s-{esc(window.get('target_full_asr_context_end_s'))}s; text-only estimate)</div><p>{esc(window.get('target_full_asr_context_text'))}</p>
  <div class="label">gold target sentence(s) assigned to this source window</div>
  <ul>{sent_html or '<li>No sentence assigned.</li>'}</ul>
</section>"""
        )
    output_path.write_text(
        f"""<!doctype html>
<meta charset="utf-8">
<title>{esc(run.get('run_id'))}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;line-height:1.45;color:#202428}}
h1{{font-size:22px;margin:0 0 8px}}h2{{font-size:15px;margin:0 0 8px}}
.meta,.label{{color:#5d6570;font-size:13px}}section{{border-top:1px solid #d6d9de;padding:16px 0}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}audio{{width:100%}}
.actions{{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}}
button{{border:1px solid #c9ced6;background:#fff;border-radius:6px;padding:4px 8px;cursor:pointer}}
.audio-tools{{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;margin-top:4px;color:#5d6570;font-size:12px}}
.audio-tools input{{width:100%}}
.source-sentence,.score{{display:block;color:#5d6570;font-size:12px}}
li.covered{{color:#116329}}li.partial{{color:#9a6700}}li.missed{{color:#b42318}}
p{{white-space:pre-wrap;overflow-wrap:anywhere}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style>
<h1>{esc(run.get('run_id'))}</h1>
<div class="meta">direction {esc(run.get('direction'))} · speed {esc(run.get('speed_factor'))} · WER {esc(metrics.get('wer'))} · CER {esc(metrics.get('cer'))} · BLEU {esc(metrics.get('bleu'))} · chrF {esc(metrics.get('chrf'))} · max backlog {esc(metrics.get('max_backlog_s'))}</div>
<section>
  <h2>Full Audio</h2>
  <div class="grid">
    <div><div class="label">source original</div><audio id="full-source-original" controls preload="metadata" src="{esc(rel(run.get('source_eval_wav_path') or run.get('source_wav_path'), run_dir))}"></audio></div>
    <div><div class="label">source streamed</div><audio id="full-source-stream" controls preload="metadata" src="{esc(rel(run.get('source_stream_wav_path') or run.get('source_eval_wav_path') or run.get('source_wav_path'), run_dir))}"></audio></div>
    <div><div class="label">generated target</div><audio id="full-target-audio" controls preload="metadata" src="{esc(rel(metrics.get('generated_wav_path') or '', run_dir))}"></audio></div>
  </div>
  <h2>Reference</h2><p>{esc(run.get('target_reference_text'))}</p>
  <h2>ASR / Output Transcript</h2><p>{esc(metrics.get('candidate_text'))}</p>
</section>
{''.join(sections)}
<script>
function fmtTime(s) {{
  if (!Number.isFinite(s)) return "0:00";
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60).toString().padStart(2, "0");
  return `${{m}}:${{r}}`;
}}
document.querySelectorAll("audio").forEach((audio, idx) => {{
  const tools = document.createElement("div");
  tools.className = "audio-tools";
  const range = document.createElement("input");
  range.type = "range";
  range.min = "0";
  range.max = "1";
  range.step = "0.001";
  range.value = "0";
  const time = document.createElement("span");
  time.textContent = "0:00 / 0:00";
  tools.append(range, time);
  audio.insertAdjacentElement("afterend", tools);
  function update() {{
    const duration = Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : 0;
    range.value = duration ? String(audio.currentTime / duration) : "0";
    time.textContent = `${{fmtTime(audio.currentTime)}} / ${{fmtTime(duration)}}`;
  }}
  range.addEventListener("input", () => {{
    if (Number.isFinite(audio.duration) && audio.duration > 0) {{
      audio.currentTime = Number(range.value) * audio.duration;
    }}
  }});
  audio.addEventListener("loadedmetadata", update);
  audio.addEventListener("timeupdate", update);
  update();
}});
document.querySelectorAll("button[data-seek-audio]").forEach((button) => {{
  button.addEventListener("click", () => {{
    const audio = document.getElementById(button.dataset.seekAudio);
    const target = Number(button.dataset.seekS || "0");
    if (audio) {{
      audio.currentTime = Math.max(0, target);
      audio.play();
    }}
  }});
}});
</script>
""",
        encoding="utf-8",
    )


def write_combined_dashboard(output_path: str | Path, metric_rows: list[dict[str, Any]]) -> None:
    output_path = Path(output_path)
    rows_html = []
    for row in metric_rows:
        rows_html.append(
            "<tr>"
            f"<td><a href='{esc(row.get('run_page_rel'))}'>{esc(row.get('run_id'))}</a></td>"
            f"<td>{esc(row.get('direction'))}</td>"
            f"<td>{esc(row.get('speed_factor'))}</td>"
            f"<td>{esc(row.get('wer'))}</td>"
            f"<td>{esc(row.get('cer'))}</td>"
            f"<td>{esc(row.get('bleu'))}</td>"
            f"<td>{esc(row.get('chrf'))}</td>"
            f"<td>{esc(row.get('max_backlog_s'))}</td>"
            f"<td>{esc(row.get('backlog_violation_rate'))}</td>"
            f"<td>{esc(row.get('missed_sentence_rate'))}</td>"
            f"<td>{esc(row.get('end_lag_s'))}</td>"
            "</tr>"
        )
    output_path.write_text(
        f"""<!doctype html>
<meta charset="utf-8">
<title>FLORAS Live S2S Dashboard</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;line-height:1.45;color:#202428}}
h1{{font-size:22px}}table{{border-collapse:collapse;width:100%}}td,th{{border-top:1px solid #d6d9de;padding:8px;text-align:left;font-size:13px}}
th{{background:#f6f8fa}}
</style>
<h1>FLORAS Live S2S Dashboard</h1>
<table>
<thead><tr><th>run</th><th>direction</th><th>speed</th><th>WER</th><th>CER</th><th>BLEU</th><th>chrF</th><th>max backlog</th><th>backlog violation</th><th>missed sentence</th><th>end lag</th></tr></thead>
<tbody>{''.join(rows_html)}</tbody>
</table>
""",
        encoding="utf-8",
    )
