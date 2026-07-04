#!/usr/bin/env python3
"""Run ACL6060 offline audio-to-text translation eval with GPT/Gemini."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
import wave
from argparse import Namespace
from pathlib import Path
from typing import Any


LANGUAGE_NAMES = {
    "zh": "Simplified Chinese",
    "ja": "Japanese",
    "de": "German",
}

DEFAULT_MODELS = {
    "openai": "gpt-audio-1.5",
    "gemini": "gemini-3.5-flash",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ACL6060 SimulEval instances.log files by translating "
            "segmented source speech with an audio-capable LLM API."
        )
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--provider", required=True, choices=sorted(DEFAULT_MODELS))
    parser.add_argument("--api-key-file", required=True, type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--split", default="dev", choices=["dev", "eval"])
    parser.add_argument("--target-lang", default="zh", choices=sorted(LANGUAGE_NAMES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--http-timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep-s", type=float, default=5.0)
    parser.add_argument("--deps-dir", type=Path, default=None)
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--sacrebleu-tokenizer", default=None)
    parser.add_argument("--eval-latency-unit", default=None, choices=["word", "char"])
    return parser.parse_args()


def read_secret(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"empty API key file: {path}")
    return value


def reference_path(dataset_root: Path, split: str, target_lang: str) -> Path:
    return dataset_root / split / "text" / "txt" / f"ACL.6060.{split}.en-xx.{target_lang}.txt"


def wav_path(dataset_root: Path, split: str, index: int) -> Path:
    return dataset_root / split / "segmented_wavs" / "gold" / f"sent_{index + 1}.wav"


def wav_duration_ms(path: Path) -> float:
    with wave.open(path.as_posix(), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
    return round(frames * 1000.0 / rate, 3)


def target_units(text: str, latency_unit: str) -> int:
    stripped = text.strip()
    if latency_unit == "char":
        return len(stripped.replace(" ", ""))
    return len(stripped.split())


def clean_prediction(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    return " ".join(text.splitlines()).strip()


def build_prompt(target_lang: str) -> str:
    language_name = LANGUAGE_NAMES[target_lang]
    return (
        "Translate the English speech in the audio into "
        f"{language_name}. Return only the translation text. "
        "Do not include explanations, markdown, or source-language transcript."
    )


def http_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1200]}") from exc


def openai_translate(
    api_key: str,
    model: str,
    prompt: str,
    audio_b64: str,
    args: argparse.Namespace,
) -> str:
    payload = {
        "model": model,
        "modalities": ["text"],
        "temperature": args.temperature,
        "max_completion_tokens": args.max_output_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                ],
            }
        ],
    }
    response = http_json(
        "https://api.openai.com/v1/chat/completions",
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        payload,
        args.http_timeout,
    )
    content = response["choices"][0]["message"].get("content", "")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def gemini_translate(
    api_key: str,
    model: str,
    prompt: str,
    audio_b64: str,
    args: argparse.Namespace,
) -> str:
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": args.temperature,
            "maxOutputTokens": args.max_output_tokens,
        },
    }
    response = http_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        {"Content-Type": "application/json"},
        payload,
        args.http_timeout,
    )
    parts = response["candidates"][0]["content"].get("parts", [])
    return "".join(part.get("text", "") for part in parts if isinstance(part, dict))


def translate_with_retries(
    api_key: str,
    prompt: str,
    audio_path: Path,
    args: argparse.Namespace,
) -> tuple[str, float]:
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    model = args.model or DEFAULT_MODELS[args.provider]
    start = time.time()
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            if args.provider == "openai":
                text = openai_translate(api_key, model, prompt, audio_b64, args)
            else:
                text = gemini_translate(api_key, model, prompt, audio_b64, args)
            return clean_prediction(text), (time.time() - start) * 1000.0
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= args.retries:
                break
            time.sleep(args.retry_sleep_s * (attempt + 1))
    raise RuntimeError(f"translation failed for {audio_path}: {last_error}") from last_error


def load_existing_indices(instances_path: Path) -> set[int]:
    if not instances_path.exists():
        return set()
    indices: set[int] = set()
    with instances_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                indices.add(json.loads(line)["index"])
    return indices


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def score_output(output_dir: Path, deps_dir: Path | None, target_lang: str, args: argparse.Namespace) -> None:
    if deps_dir is not None:
        sys.path.insert(0, deps_dir.as_posix())
    try:
        from simuleval.evaluator import SentenceLevelEvaluator
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "SimulEval is required for scoring. Pass --deps-dir pointing to a "
            "directory containing simuleval and sacrebleu."
        ) from exc

    latency_unit = args.eval_latency_unit or ("word" if target_lang == "de" else "char")
    tokenizer = args.sacrebleu_tokenizer or ("13a" if target_lang == "de" else "zh")
    with (output_dir / "instances.log").open(encoding="utf-8") as f:
        first_index = 0
        for line in f:
            if line.strip():
                first_index = json.loads(line)["index"]
                break
    simuleval_args = Namespace(
        output=output_dir.as_posix(),
        score_only=True,
        no_scoring=False,
        no_progress_bar=True,
        source_type="speech",
        target_type="text",
        source_segment_size=10,
        start_index=first_index,
        end_index=-1,
        quality_metrics=["BLEU"],
        latency_metrics=["LAAL", "AL", "AP", "DAL", "ATD"],
        computation_aware=False,
        no_use_ref_len=False,
        eval_latency_unit=latency_unit,
        eval_latency_spm_model=None,
        sacrebleu_tokenizer=tokenizer,
    )
    evaluator = SentenceLevelEvaluator.from_args(simuleval_args)
    evaluator.dump_results()
    evaluator.dump_metrics()


def main() -> None:
    args = parse_args()
    model = args.model or DEFAULT_MODELS[args.provider]
    refs_path = reference_path(args.dataset_root, args.split, args.target_lang)
    references = refs_path.read_text(encoding="utf-8").splitlines()
    end_index = len(references)
    if args.limit is not None:
        end_index = min(end_index, args.start_index + args.limit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    instances_path = args.output_dir / "instances.log"
    responses_path = args.output_dir / "responses.jsonl"
    if not args.resume:
        instances_path.write_text("", encoding="utf-8")
        responses_path.write_text("", encoding="utf-8")

    api_key = read_secret(args.api_key_file)
    prompt = build_prompt(args.target_lang)
    latency_unit = args.eval_latency_unit or ("word" if args.target_lang == "de" else "char")
    done = load_existing_indices(instances_path) if args.resume else set()

    metadata = {
        "provider": args.provider,
        "model": model,
        "split": args.split,
        "target_lang": args.target_lang,
        "dataset_root": args.dataset_root.as_posix(),
        "reference_path": refs_path.as_posix(),
        "start_index": args.start_index,
        "end_index": end_index,
        "latency_unit": latency_unit,
        "sacrebleu_tokenizer": args.sacrebleu_tokenizer
        or ("13a" if args.target_lang == "de" else "zh"),
        "prompt": prompt,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for index in range(args.start_index, end_index):
        if index in done:
            continue
        source_wav = wav_path(args.dataset_root, args.split, index)
        source_length = wav_duration_ms(source_wav)
        error = None
        try:
            prediction, elapsed_ms = translate_with_retries(api_key, prompt, source_wav, args)
        except Exception as exc:  # noqa: BLE001
            if args.stop_on_error:
                raise
            prediction = ""
            elapsed_ms = 0.0
            error = str(exc)
        unit_count = target_units(prediction, latency_unit)
        delays = [source_length] * max(unit_count, 1)
        elapsed = [source_length + elapsed_ms] * max(unit_count, 1)
        record = {
            "index": index,
            "prediction": prediction,
            "delays": delays,
            "elapsed": elapsed,
            "prediction_length": unit_count,
            "reference": references[index],
            "source": [source_wav.as_posix()],
            "source_length": source_length,
        }
        if error is not None:
            record["error"] = error
        write_jsonl(instances_path, record)
        response_record = {
            "index": index,
            "provider": args.provider,
            "model": model,
            "prediction": prediction,
            "elapsed_ms": round(elapsed_ms, 3),
        }
        if error is not None:
            response_record["error"] = error
        write_jsonl(responses_path, response_record)
        progress_record = {
            "index": index,
            "prediction_length": unit_count,
            "elapsed_ms": round(elapsed_ms, 3),
        }
        if error is not None:
            progress_record["error"] = error[:240]
        print(json.dumps(progress_record, ensure_ascii=False), flush=True)

    if not args.no_score:
        score_output(args.output_dir, args.deps_dir, args.target_lang, args)


if __name__ == "__main__":
    main()
