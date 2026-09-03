#!/usr/bin/env python3
"""First-cut scoring for the InfiniSST + MOSS v2 cascade on ACL6060.

Per (talk, chunk-size) run: concatenate session wavs into one talk-level
target wav, ASR it (whisper large-v3 zh unless an OpenAI key file is given,
then gpt-4o-mini-transcribe windows), and score BLEU/chrF (sacrebleu,
tokenize=zh) against the talk reference plus duration/backlog stats.

# note (luojiaxuan): corpus BLEU over full-talk text is an approximation of
# the canonical SEGALE-aligned BLEU; label results accordingly. XCOMET-XL and
# LongYAAL run separately via the kit-lecture-translator pipeline.
"""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", required=True)
    parser.add_argument("--target-list", required=True, help="acl_zh target.list (one talk reference per line)")
    parser.add_argument("--source-order", default="268,367,590,110,117", help="talk order of target.list lines")
    parser.add_argument("--talk-wav-dir", required=True, help="dir with 2022.acl-long.<talk>.wav sources")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--openai-key-file", default=None,
                        help="use gpt-4o-mini-transcribe windowed ASR (canonical) instead of whisper")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def concat_run(bench: Path, talk: int, chunk: str) -> tuple[Path, float]:
    rows = [
        json.loads(line)
        for line in (bench / "tts_rows" / f"talk{talk}.chunk{chunk}.rows.jsonl").open(encoding="utf-8")
    ]
    out_path = bench / "tts_wavs" / f"talk{talk}.chunk{chunk}.full.wav"
    summaries = {
        json.loads(line)["row_id"]: json.loads(line)
        for line in (bench / "tts_wavs" / f"talk{talk}.chunk{chunk}.summary.jsonl").open(encoding="utf-8")
    }
    pcm, rate = [], None
    for row in rows:
        rec = summaries.get(row["row_id"])
        if rec is None or rec.get("failure") or not rec.get("wav"):
            continue
        with wave.open(rec["wav"], "rb") as handle:
            rate = rate or handle.getframerate()
            pcm.append(handle.readframes(handle.getnframes()))
    with wave.open(str(out_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        for chunk_pcm in pcm:
            handle.writeframes(chunk_pcm)
    duration = sum(len(c) for c in pcm) / 2 / rate
    return out_path, duration


def transcribe_elevenlabs(
    wav_path: Path,
    key_file: str = "~/.keys/elevenlabs_sst_data",
    model_id: str = "scribe_v2",
    language_code: str = "zh",
    endpoint: str = "https://api.elevenlabs.io/v1/speech-to-text",
    raw_out: Path | None = None,
    timeout_s: float = 900.0,
) -> str:
    """One request over the WHOLE wav to ElevenLabs Scribe (files up to 10 h; no
    windowing, no turn boundaries in the transcript). Request shape follows
    Open-LiveTranslate's scorer: multipart file + model_id + word timestamps, key in
    the xi-api-key header, six retries on 429/5xx. The raw answer (text + word
    timestamps) is written to raw_out when given."""
    import os
    import time
    import urllib.error
    import urllib.request
    import uuid

    key = next(tok.split()[0] for tok in Path(os.path.expanduser(key_file))
               .read_text(encoding="utf-8").splitlines()
               if tok.strip() and not tok.strip().startswith("#"))
    boundary = "----s2s" + uuid.uuid4().hex
    fields = [("model_id", model_id), ("timestamps_granularity", "word"),
              ("diarize", "false"), ("tag_audio_events", "false"),
              ("language_code", language_code)]
    parts = [f"--{boundary}\r\nContent-Disposition: form-data; name=\"{n}\"\r\n\r\n{v}\r\n"
             .encode("utf-8") for n, v in fields]
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                 f"filename=\"{wav_path.name}\"\r\nContent-Type: audio/wav\r\n\r\n".encode("utf-8")
                 + wav_path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)
    for attempt, sleep_s in enumerate((2, 4, 8, 16, 32, 64, None)):
        req = urllib.request.Request(endpoint, data=body, method="POST")
        req.add_header("xi-api-key", key)
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                answer = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if sleep_s is None or not (exc.code == 429 or exc.code >= 500):
                raise SystemExit(f"ElevenLabs HTTP {exc.code} for {wav_path.name}: {detail}") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            if sleep_s is None:
                raise SystemExit(f"ElevenLabs transport error for {wav_path.name}: {exc}") from None
        time.sleep(sleep_s)
    answer["_request_elapsed_s"] = round(time.monotonic() - started, 3)
    if raw_out is not None:
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        raw_out.write_text(json.dumps(answer, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return str(answer.get("text") or "").strip()


def transcribe_openai_windows(
    wav_path: Path,
    key: str | None,
    window_s: float = 120.0,
    base_url: str = "https://api.openai.com",
    model: str = "gpt-4o-mini-transcribe",
) -> str:
    """Windowed ASR over <=120s chunks against any OpenAI-compatible
    transcriptions endpoint (OpenAI cloud or a self-hosted Qwen3-ASR)."""
    import io
    import urllib.request

    with wave.open(str(wav_path), "rb") as handle:
        rate = handle.getframerate()
        pcm = handle.readframes(handle.getnframes())
    window_bytes = int(window_s * rate) * 2
    texts = []
    for start in range(0, len(pcm), window_bytes):
        # note (luojiaxuan): 近静音窗直接记空串，不发请求。OpenAI baseline 的
        # target 语音稀疏，存在整窗静音；sglang 的 Qwen3-ASR 对零特征音频会
        # 崩掉整个 scheduler（Insufficient multimodal embedding length），
        # 一个静音窗能带死服务并让后续所有请求 RemoteDisconnected。
        # 静音转写为空本来就是语义正确的行为。阈值 30 ≈ int16 满量程 0.1%。
        import array as _array
        _samples = _array.array("h")
        _samples.frombytes(pcm[start : start + window_bytes])
        _sub = _samples[::16] or _array.array("h", [0])
        if (sum(v * v for v in _sub) / len(_sub)) ** 0.5 < 30.0:
            texts.append("")
            continue
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm[start : start + window_bytes])
        body = buf.getvalue()
        boundary = "----acl6060cascade"
        parts = []
        for name, value in (("model", model), ("language", "zh")):
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
            )
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"win.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode()
            + body
            + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode())
        data = b"".join(parts)
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(
            base_url.rstrip("/") + "/v1/audio/transcriptions",
            data=data,
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            texts.append(json.loads(resp.read())["text"].strip())
    return "".join(texts)


def main() -> None:
    args = parse_args()
    import sacrebleu

    bench = Path(args.bench_dir)
    talk_order = [int(t) for t in args.source_order.split(",")]
    refs_by_talk = {}
    with Path(args.target_list).open(encoding="utf-8") as handle:
        for talk, line in zip(talk_order, handle):
            refs_by_talk[talk] = line.strip()

    openai_key = None
    if args.openai_key_file:
        openai_key = Path(args.openai_key_file).read_text().strip()
        model = None
        asr_label = "gpt-4o-mini-transcribe-120swin"
    else:
        import whisper

        model = whisper.load_model(args.whisper_model, device=args.device)
        asr_label = f"whisper-{args.whisper_model}"
    results = []
    for chunk in ("096", "192"):
        hyps, refs = [], []
        rows_out = []
        for talk in (110, 117, 268, 367, 590):
            wav_path, target_s = concat_run(bench, talk, chunk)
            with wave.open(str(Path(args.talk_wav_dir) / f"2022.acl-long.{talk}.wav"), "rb") as handle:
                source_s = handle.getnframes() / handle.getframerate()
            if openai_key:
                hyp = transcribe_openai_windows(wav_path, openai_key)
            else:
                asr = model.transcribe(str(wav_path), language="zh", temperature=0.0)
                hyp = asr["text"].strip()
            hyps.append(hyp)
            refs.append(refs_by_talk[talk])
            rows_out.append(
                {
                    "talk": talk,
                    "chunk": chunk,
                    "source_s": round(source_s, 2),
                    "target_s": round(target_s, 2),
                    "duration_ratio": round(target_s / source_s, 4),
                    "hyp_chars": len(hyp),
                }
            )
            print(json.dumps(rows_out[-1]), flush=True)
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh")
        chrf = sacrebleu.corpus_chrf(hyps, [refs])
        results.append(
            {
                "system": f"infinisst_mossv2_chunk{chunk}",
                "asr": asr_label,
                "bleu_zh_approx": round(bleu.score, 2),
                "chrf_approx": round(chrf.score, 2),
                "talks": rows_out,
                "duration_ratio_mean": round(
                    sum(r["duration_ratio"] for r in rows_out) / len(rows_out), 4
                ),
            }
        )
    Path(args.output_json).write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
