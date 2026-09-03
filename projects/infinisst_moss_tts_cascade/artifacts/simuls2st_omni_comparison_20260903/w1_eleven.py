#!/usr/bin/env python3
"""ElevenLabs Scribe v2 transcription of each talk's whole rendered target wav, in the
shape w3/w4/w5 consume (instances.log with one prediction per document).

One HTTP request per talk over the complete audio (Scribe v2 accepts files up to 10 h;
files longer than 8 min are parallelised server-side), so no windowing and no turn
boundaries enter the transcript. The request mirrors Open-LiveTranslate's scoring
pre-step (eval/scripts/asr_bench/transcribe_elevenlabs.py): multipart file + model_id +
timestamps_granularity=word + diarize=false + tag_audio_events=false + language_code,
key in the xi-api-key header, six retries on 429/5xx with 2..64 s backoff. The raw
answer (text + word timestamps) is kept per talk for later latency work.

  w1_eleven.py --ext-dir <dir with instances.log + wavs/> --run-dir <out> \\
      --dataset-root /data/ap_score/d0_3t --label ours_v8 --key-file ~/.keys/elevenlabs
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

TALKS = (268, 110, 117)
DEFAULT_ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"
RETRY_SLEEPS = (2, 4, 8, 16, 32, 64)


def read_key(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if token and not token.startswith("#"):
            return token.split()[0]
    raise SystemExit(f"no key in {path}")


def encode_multipart(fields: list[tuple[str, str]], file_field: str, filename: str,
                     payload: bytes) -> tuple[bytes, str]:
    boundary = "----s2s" + uuid.uuid4().hex
    parts = []
    for name, value in fields:
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                     f"{value}\r\n".encode("utf-8"))
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
                 f"filename=\"{filename}\"\r\nContent-Type: audio/wav\r\n\r\n".encode("utf-8")
                 + payload + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def transcribe(wav_path: Path, api_key: str, endpoint: str, model_id: str,
               language_code: str, timeout_s: float) -> dict:
    payload = wav_path.read_bytes()
    fields = [("model_id", model_id), ("timestamps_granularity", "word"),
              ("diarize", "false"), ("tag_audio_events", "false"),
              ("language_code", language_code)]
    body, content_type = encode_multipart(fields, "file", wav_path.name, payload)
    attempt = 0
    while True:
        request = urllib.request.Request(endpoint, data=body, method="POST")
        request.add_header("xi-api-key", api_key)
        request.add_header("Content-Type", content_type)
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                answer = json.loads(response.read().decode("utf-8"))
            answer["_request_elapsed_s"] = round(time.monotonic() - started, 3)
            return answer
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            retryable = exc.code == 429 or exc.code >= 500
            if not retryable or attempt >= len(RETRY_SLEEPS):
                raise SystemExit(f"[FATAL] ElevenLabs HTTP {exc.code} for {wav_path.name}: "
                                 f"{detail}") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt >= len(RETRY_SLEEPS):
                raise SystemExit(f"[FATAL] ElevenLabs transport error for {wav_path.name}: "
                                 f"{exc}") from None
        time.sleep(RETRY_SLEEPS[attempt])
        attempt += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ext-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--key-file", default=os.environ.get("ELEVENLABS_KEY_FILE",
                                                             "~/.keys/elevenlabs_sst_data"))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model-id", default="scribe_v2")
    parser.add_argument("--language-code", default="zh")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    api_key = read_key(Path(os.path.expanduser(args.key_file)))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.run_dir / "elevenlabs_raw"
    raw_dir.mkdir(exist_ok=True)
    inputs = args.dataset_root / "main_result/inputs/acl_zh"
    config = {
        "provider": f"external_{args.label}",
        "target_lang": "zh", "lang_code": "zh", "speed_factor": 1.0, "chunk_ms": 2000,
        "dataset_root": str(args.dataset_root),
        "source_text_file": str(inputs / "source_text.txt"),
        "ref_file": str(inputs / "ref.txt"),
        "audio_yaml": str(inputs / "audio.yaml"),
        "asr": f"elevenlabs-{args.model_id} whole-talk single request, language_code={args.language_code}",
        "asr_endpoint": args.endpoint,
        "timing_protocol": "uniform_proxy_NOT_comparable",
        "tts": f"{args.label}: rendered target speech, one wav per talk",
    }
    (args.run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    ext_rows = {}
    for line in (args.ext_dir / "instances.log").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            ext_rows[Path(row["source"]).stem] = row

    rows_out, qa = [], []
    for index, talk in enumerate(TALKS):
        row = ext_rows[f"2022.acl-long.{talk}"]
        wav_path = args.ext_dir / "wavs" / f"{row['index']}_pred.wav"
        raw_path = raw_dir / f"talk{talk}.json"
        if raw_path.exists():
            answer = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            answer = transcribe(wav_path, api_key, args.endpoint, args.model_id,
                                args.language_code, args.timeout)
            raw_path.write_text(json.dumps(answer, ensure_ascii=False, indent=1) + "\n",
                                encoding="utf-8")
        prediction = str(answer.get("text") or "").strip()
        words = [w for w in (answer.get("words") or []) if w.get("type") == "word"]
        source_wav = args.dataset_root / f"main_result/audio/acl6060/2022.acl-long.{talk}.wav"
        with wave.open(str(source_wav), "rb") as handle:
            source_ms = handle.getnframes() / handle.getframerate() * 1000.0
        with wave.open(str(wav_path), "rb") as handle:
            target_s = handle.getnframes() / handle.getframerate()
        units = [char for char in prediction if not char.isspace()]
        count = max(1, len(units))
        delays = [round((unit + 1) / count * source_ms, 3) for unit in range(len(units))]
        rows_out.append({"index": index, "source": [str(source_wav)], "prediction": prediction,
                         "delays": delays, "elapsed": delays})
        qa.append({"talk": talk, "chars": len(units), "words": len(words),
                   "target_s": round(target_s, 3), "source_s": round(source_ms / 1000.0, 3),
                   "language_probability": answer.get("language_probability"),
                   "request_elapsed_s": answer.get("_request_elapsed_s")})
        print(json.dumps(qa[-1], ensure_ascii=False), flush=True)

    with (args.run_dir / "instances.log").open("w", encoding="utf-8") as handle:
        for out in rows_out:
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
    (args.run_dir / "session_qa.json").write_text(json.dumps({"per_talk": qa}, indent=2) + "\n")
    print(f"ELEVEN_RUNDIR_DONE {args.run_dir}", flush=True)


if __name__ == "__main__":
    main()
