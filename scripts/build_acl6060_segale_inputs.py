#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

CHAR_LEVEL_LANGS = {"zh", "ja"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ACL6060 inputs for SEGALE alignment.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def resolve_path(value: Any, dataset_root: Path | None = None) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    if dataset_root is not None:
        candidates = [dataset_root / path]
        if "main_result/" in path.as_posix():
            suffix = path.as_posix().split("main_result/", 1)[1]
            candidates.append(dataset_root / "main_result" / suffix)
        for candidate in candidates:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(path)


def doc_id(value: Any) -> str:
    return Path(str(value).replace("\\", "/")).name


def prediction_units_with_spans(
    text: str, target_lang: str
) -> tuple[list[str], list[tuple[int, int]]]:
    if target_lang in CHAR_LEVEL_LANGS:
        units = []
        spans = []
        for index, char in enumerate(text):
            if char.isspace():
                continue
            units.append(char)
            spans.append((index, index + 1))
        return units, spans
    matches = list(re.finditer(r"\S+", text))
    return [match.group(0) for match in matches], [
        (match.start(), match.end()) for match in matches
    ]


def build_segale_inputs(
    run_dir: Path,
    output_dir: Path,
    dataset_root_override: Path | None = None,
) -> dict[str, Any]:
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    dataset_root = dataset_root_override
    if dataset_root is None and config.get("dataset_root"):
        dataset_root = Path(str(config["dataset_root"]))
    source_path = resolve_path(config["source_text_file"], dataset_root)
    reference_path = resolve_path(config["ref_file"], dataset_root)
    yaml_path = resolve_path(config["audio_yaml"], dataset_root)
    source_lines = source_path.read_text(encoding="utf-8").splitlines()
    reference_lines = reference_path.read_text(encoding="utf-8").splitlines()
    yaml_rows = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or []
    if not (len(source_lines) == len(reference_lines) == len(yaml_rows)):
        raise ValueError(
            "source/reference/yaml length mismatch: "
            f"{len(source_lines)} vs {len(reference_lines)} vs {len(yaml_rows)}"
        )

    ordered_docs = list(dict.fromkeys(doc_id(row["wav"]) for row in yaml_rows))
    instances = sorted(read_jsonl(run_dir / "instances.log"), key=lambda row: int(row["index"]))
    if len(instances) != len(ordered_docs):
        raise ValueError(f"instances/doc mismatch: {len(instances)} vs {len(ordered_docs)}")

    source_by_doc: dict[str, list[str]] = {key: [] for key in ordered_docs}
    ref_rows = []
    normalized_yaml = []
    speed_factor = float(config.get("speed_factor") or 1.0)
    for index, (source, reference, segment) in enumerate(
        zip(source_lines, reference_lines, yaml_rows)
    ):
        key = doc_id(segment["wav"])
        source_by_doc[key].append(source)
        ref_rows.append(
            {
                "src": source,
                "tgt": reference,
                "sys_id": "acl6060_reference",
                "doc_id": key,
                # note (luojiaxuan): Speech-to-Speech-Latency expects the
                # original global source index to be 1-based in `src_ref_ids`.
                # Its LongYAAL matcher converts it back to a 0-based YAML row.
                "seg_id": index + 1,
            }
        )
        normalized = dict(segment)
        normalized["wav"] = key
        if not math.isclose(speed_factor, 1.0):
            normalized["offset"] = float(normalized.get("offset") or 0.0) / speed_factor
            normalized["duration"] = float(normalized.get("duration") or 0.0) / speed_factor
        normalized_yaml.append(normalized)

    target_lang = str(config.get("target_lang") or "")
    hyp_rows = []
    latency_instances = []
    for expected_index, (key, instance) in enumerate(zip(ordered_docs, instances)):
        if int(instance["index"]) != expected_index:
            raise ValueError(f"unexpected instance index {instance['index']} at {expected_index}")
        source_value = instance.get("source")
        source_item = (
            source_value[0] if isinstance(source_value, list) and source_value else source_value
        )
        if doc_id(source_item) != key:
            raise ValueError(f"instance/doc mismatch: {doc_id(source_item)} vs {key}")
        prediction = str(instance.get("prediction") or "")
        units, spans = prediction_units_with_spans(prediction, target_lang)
        delays = list(instance.get("delays") or [])
        elapsed = list(instance.get("elapsed") or delays)
        if len(units) != len(delays) or len(units) != len(elapsed):
            raise ValueError(
                f"{key} target units/timestamps mismatch: "
                f"{len(units)} vs {len(delays)} vs {len(elapsed)}"
            )
        hyp_rows.append(
            {
                "src": " ".join(source_by_doc[key]),
                "tgt": prediction,
                "sys_id": run_dir.name,
                "doc_id": key,
                "seg_id": expected_index,
            }
        )
        latency_row = dict(instance)
        latency_row["prediction_text"] = prediction
        latency_row["prediction_units"] = units
        latency_row["prediction_unit_char_starts_ends"] = spans
        latency_row["durations"] = list(instance.get("durations") or [])
        latency_instances.append(latency_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "ref.jsonl", ref_rows)
    write_jsonl(output_dir / "hyp.jsonl", hyp_rows)
    write_jsonl(output_dir / "instances.segale.jsonl", latency_instances)
    (output_dir / "audio.scaled.basename.yaml").write_text(
        yaml.safe_dump(normalized_yaml, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    summary = {
        "alignment_backend": "SEGALE",
        "run_dir": str(run_dir),
        "target_lang": target_lang,
        "speed_factor": speed_factor,
        "documents": len(ordered_docs),
        "source_segments": len(ref_rows),
        "hypothesis_documents": len(hyp_rows),
        "ref_jsonl": str(output_dir / "ref.jsonl"),
        "hyp_jsonl": str(output_dir / "hyp.jsonl"),
        "latency_instances_jsonl": str(output_dir / "instances.segale.jsonl"),
        "scaled_yaml": str(output_dir / "audio.scaled.basename.yaml"),
    }
    (output_dir / "input_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.run_dir / "segale_alignment")
    print(
        json.dumps(
            build_segale_inputs(args.run_dir, output_dir, args.dataset_root),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
