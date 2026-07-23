#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


TARGET_LANG_TO_DISPLAY = {"zh": "En-Zh", "de": "En-De", "ja": "En-Ja"}
CHAR_LEVEL_LANGS = {"zh", "ja"}
BLEU_TOKENIZER_BY_LANG = {"zh": "zh", "ja": "ja-mecab", "de": "13a"}
OMNISTEVAL_LANG_BY_LANG = {"zh": "", "ja": "", "de": "de"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OmniSTEval LongYAAL/BLEU over an ACL6060 full-wav instances.log."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--target-lang", choices=sorted(TARGET_LANG_TO_DISPLAY), default="")
    parser.add_argument("--speed-factor", type=float, default=0.0)
    parser.add_argument("--omnisteval-bin", default="")
    parser.add_argument("--bleu-tokenizer", default="")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def read_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_from_config(path_value: Any, dataset_root: Path | None = None) -> Path:
    path = Path(str(path_value))
    if path.exists():
        return path
    if dataset_root is not None:
        candidate = dataset_root / path.as_posix()
        if candidate.exists():
            return candidate
        if "main_result/" in path.as_posix():
            rel = path.as_posix().split("main_result/", 1)[1]
            candidate = dataset_root / "main_result" / rel
            if candidate.exists():
                return candidate
    raise FileNotFoundError(str(path))


def normalized_prediction_and_units(
    text: str,
    target_lang: str,
) -> tuple[str, list[str]]:
    if target_lang in CHAR_LEVEL_LANGS:
        units = [ch for ch in text if not ch.isspace()]
        return "".join(units), units
    units = text.split()
    return " ".join(units), units


def fit_emissions(values: list[Any], expected: int) -> tuple[list[float], str]:
    out = [float(value) for value in values]
    if len(out) == expected:
        return out, ""
    if expected <= 0:
        return [], f"drop_{len(out)}_emissions_for_empty_prediction"
    if not out:
        return [0.0] * expected, f"pad_empty_emissions_to_{expected}"
    if len(out) > expected:
        return out[:expected], f"trim_emissions_{len(out)}_to_{expected}"
    return out + [out[-1]] * (expected - len(out)), f"pad_emissions_{len(out)}_to_{expected}"


def write_normalized_hypothesis(
    instances_path: Path,
    output_path: Path,
    target_lang: str,
) -> dict[str, Any]:
    rows = []
    warnings = []
    for row in read_jsonl(instances_path):
        prediction, units = normalized_prediction_and_units(str(row.get("prediction") or ""), target_lang)
        out = dict(row)
        out["prediction"] = prediction
        out["prediction_length"] = len(units)
        for source_key, target_key in [("delays", "delays"), ("elapsed", "elapsed")]:
            if source_key not in row:
                continue
            fitted, note = fit_emissions(list(row.get(source_key) or []), len(units))
            out[target_key] = fitted
            if note:
                warnings.append(
                    {
                        "index": row.get("index"),
                        "field": source_key,
                        "note": note,
                        "original_len": len(row.get(source_key) or []),
                        "normalized_units": len(units),
                    }
                )
        rows.append(out)
    write_jsonl(output_path, rows)
    if warnings:
        write_jsonl(output_path.with_suffix(".warnings.jsonl"), warnings)
    return {
        "rows": len(rows),
        "warnings": len(warnings),
        "hypothesis": str(output_path),
        "warnings_path": str(output_path.with_suffix(".warnings.jsonl")) if warnings else "",
    }


def load_yaml(path: Path) -> Any:
    if yaml is None:
        raise RuntimeError("PyYAML is required for ACL6060 audio.yaml scaling")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump_yaml(path: Path, value: Any) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required for ACL6060 audio.yaml scaling")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def scale_audio_yaml(input_path: Path, output_path: Path, speed_factor: float) -> dict[str, Any]:
    rows = load_yaml(input_path)
    if not isinstance(rows, list):
        raise ValueError(f"{input_path} must contain a list")
    scaled = []
    for row in rows:
        out = dict(row)
        if speed_factor > 0 and not math.isclose(speed_factor, 1.0):
            out["offset"] = float(out.get("offset") or 0.0) / speed_factor
            out["duration"] = float(out.get("duration") or 0.0) / speed_factor
        scaled.append(out)
    dump_yaml(output_path, scaled)
    return {"rows": len(scaled), "speech_segmentation": str(output_path)}


def read_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    if not path.exists():
        return scores
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("metric\t"):
            continue
        key, value = line.split("\t", 1)
        try:
            scores[key] = float(value)
        except ValueError:
            continue
    return scores


def endpoint_offset_summary(instances_path: Path) -> dict[str, Any]:
    delay_offsets = []
    elapsed_offsets = []
    for row in read_jsonl(instances_path):
        source_length = float(row.get("source_length") or 0.0)
        delays = row.get("delays") or []
        elapsed = row.get("elapsed") or []
        if delays:
            delay_offsets.append(float(delays[-1]) - source_length)
        if elapsed:
            elapsed_offsets.append(float(elapsed[-1]) - source_length)
    return {
        "ending_offset_cu_ms_mean": mean(delay_offsets),
        "ending_offset_ca_ms_mean": mean(elapsed_offsets),
        "ending_offset_rows": len(delay_offsets),
    }


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def resolve_omnisteval_bin(value: str) -> str:
    if value:
        return value
    found = shutil.which("omnisteval")
    if found:
        return found
    raise FileNotFoundError("omnisteval not found; install omnisteval==0.1.10 or pass --omnisteval-bin")


def run_omnisteval(
    *,
    omnisteval_bin: str,
    hypothesis: Path,
    speech_segmentation: Path,
    ref_sentences: Path,
    output_dir: Path,
    target_lang: str,
    bleu_tokenizer: str,
) -> None:
    cmd = [
        omnisteval_bin,
        "longform",
        "--speech_segmentation",
        str(speech_segmentation),
        "--ref_sentences_file",
        str(ref_sentences),
        "--hypothesis_file",
        str(hypothesis),
        "--hypothesis_format",
        "jsonl",
        "--output_folder",
        str(output_dir),
        "--bleu_tokenizer",
        bleu_tokenizer,
    ]
    if target_lang in CHAR_LEVEL_LANGS:
        cmd.append("--char_level")
    lang = OMNISTEVAL_LANG_BY_LANG[target_lang]
    if lang:
        cmd.extend(["--lang", lang])
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    config = read_config(args.run_dir)
    target_lang = args.target_lang or str(config.get("target_lang") or "zh")
    speed_factor = args.speed_factor or float(config.get("speed_factor") or 1.0)
    output_dir = args.output_dir or (args.run_dir / "omnisteval_longform")
    if args.resume and (output_dir / "summary.json").exists() and (output_dir / "scores.tsv").exists():
        scores = read_scores(output_dir / "scores.tsv")
        print(json.dumps({"skip_existing": str(output_dir), "scores": scores}, ensure_ascii=False, indent=2))
        return

    dataset_root = args.dataset_root or Path(str(config.get("dataset_root") or ""))
    instances_path = args.run_dir / "instances.log"
    if not instances_path.exists():
        raise FileNotFoundError(instances_path)

    ref_file = resolve_from_config(config.get("ref_file"), dataset_root)
    audio_yaml = resolve_from_config(config.get("audio_yaml"), dataset_root)
    hypothesis_path = output_dir / "hypothesis.omnisteval.jsonl"
    scaled_audio_yaml = output_dir / "audio.scaled.yaml"
    norm = write_normalized_hypothesis(instances_path, hypothesis_path, target_lang)
    seg = scale_audio_yaml(audio_yaml, scaled_audio_yaml, speed_factor)
    bleu_tokenizer = args.bleu_tokenizer or BLEU_TOKENIZER_BY_LANG[target_lang]
    run_omnisteval(
        omnisteval_bin=resolve_omnisteval_bin(args.omnisteval_bin),
        hypothesis=hypothesis_path,
        speech_segmentation=scaled_audio_yaml,
        ref_sentences=ref_file,
        output_dir=output_dir,
        target_lang=target_lang,
        bleu_tokenizer=bleu_tokenizer,
    )
    scores = read_scores(output_dir / "scores.tsv")
    summary = {
        "run_dir": str(args.run_dir),
        "output_dir": str(output_dir),
        "target_lang": target_lang,
        "language": TARGET_LANG_TO_DISPLAY[target_lang],
        "speed_factor": speed_factor,
        "bleu_tokenizer": bleu_tokenizer,
        "char_level": target_lang in CHAR_LEVEL_LANGS,
        "scores": scores,
        **norm,
        **seg,
        **endpoint_offset_summary(instances_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
