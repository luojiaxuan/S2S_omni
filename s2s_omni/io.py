from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .schema import S2SSample


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


def read_samples(path: str | Path) -> list[S2SSample]:
    return [S2SSample.from_dict(record) for record in read_jsonl(path)]


def write_samples(path: str | Path, samples: Iterable[S2SSample]) -> None:
    write_jsonl(path, (sample.to_dict() for sample in samples))


def read_yaml(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        data = _read_simple_yaml(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_simple_yaml(text: str) -> dict[str, Any]:
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, raw.strip()))
    if not lines:
        return {}
    value, index = _parse_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise ValueError("could not parse entire YAML file")
    if not isinstance(value, dict):
        raise ValueError("top-level YAML must be a mapping")
    return value


def _parse_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index
    is_list = lines[index][1].startswith("- ")
    if is_list:
        out: list[Any] = []
        while index < len(lines):
            line_indent, content = lines[index]
            if line_indent < indent:
                break
            if line_indent != indent or not content.startswith("- "):
                break
            item = content[2:].strip()
            index += 1
            if item:
                out.append(_parse_scalar(item))
            elif index < len(lines):
                child, index = _parse_block(lines, index, lines[index][0])
                out.append(child)
            else:
                out.append(None)
        return out, index

    out_dict: dict[str, Any] = {}
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent != indent:
            raise ValueError(f"unexpected indentation near: {content}")
        if ":" not in content:
            raise ValueError(f"expected key/value line, got: {content}")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        index += 1
        if raw_value:
            out_dict[key] = _parse_scalar(raw_value)
        elif index < len(lines) and lines[index][0] > indent:
            out_dict[key], index = _parse_block(lines, index, lines[index][0])
        else:
            out_dict[key] = {}
    return out_dict, index


def _parse_scalar(value: str) -> Any:
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if "." not in value:
            return int(value)
        return float(value)
    except ValueError:
        return value
