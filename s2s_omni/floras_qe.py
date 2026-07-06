from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


CHUNK_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)s")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def canonical_model(value: Any) -> str:
    text = str(value or "").lower()
    if "kit" in text:
        return "kit"
    if "seed" in text:
        return "seed"
    if "gemini" in text:
        return "gemini"
    if "openai" in text or "chatgpt" in text or "gpt" in text:
        return "chatgpt"
    return str(value or "")


def row_model(row: dict[str, Any]) -> str:
    return canonical_model(
        row.get("display_model")
        or row.get("compare_backend")
        or row.get("backend")
        or row.get("model")
        or row.get("eval_label")
        or ""
    )


def row_chunk_ms(row: dict[str, Any]) -> int | None:
    for key in ("compare_chunk_ms", "chunk_ms"):
        if row.get(key) is not None:
            try:
                return int(row[key])
            except (TypeError, ValueError):
                pass
    match = CHUNK_RE.search(str(row.get("display_chunk") or ""))
    if match:
        return int(round(float(match.group(1)) * 1000.0))
    return None


def row_key(row: dict[str, Any]) -> str:
    model = row_model(row)
    chunk_ms = row_chunk_ms(row)
    return "||".join(
        [
            str(row.get("run_id") or ""),
            model,
            "" if chunk_ms is None else str(chunk_ms),
            canonical_eval_label(row.get("eval_label"), model, chunk_ms),
        ]
    )


def normalized_text_len(value: Any) -> int:
    return len(" ".join(str(value or "").split()))


def canonical_eval_label(value: Any, model: str, chunk_ms: int | None) -> str:
    label = str(value or "")
    if chunk_ms is None:
        return label
    text = label.lower()
    if model == "chatgpt" and text in {
        f"openai_{chunk_ms}",
        f"openai_chunk{chunk_ms}",
        f"chatgpt_{chunk_ms}",
        f"chatgpt_chunk{chunk_ms}",
    }:
        return f"openai_{chunk_ms}"
    if model == "gemini" and text in {f"gemini_{chunk_ms}", f"gemini_chunk{chunk_ms}"}:
        return f"gemini_{chunk_ms}"
    if model == "seed" and text in {
        f"seed_{chunk_ms}",
        f"seed_chunk{chunk_ms}",
        f"seed_ast_{chunk_ms}",
        f"seed_ast_chunk{chunk_ms}",
    }:
        return f"seed_{chunk_ms}"
    if model == "kit" and text == f"kit_{chunk_ms}":
        return f"{model}_{chunk_ms}"
    return label


def load_qe_scores(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    scores: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = str(row.get("qe_row_key") or row.get("row_key") or "")
        if key:
            scores[key] = row
        scores[row_key(row)] = row
    return scores


def attach_qe_scores(row: dict[str, Any], scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not scores:
        return row
    match = scores.get(row_key(row))
    if not match:
        return row
    out = dict(row)
    expected_hypothesis_chars = match.get("hypothesis_chars")
    hypothesis_text = row.get("candidate_text") or row.get("hypothesis_text")
    if expected_hypothesis_chars is not None and hypothesis_text is not None:
        actual_hypothesis_chars = normalized_text_len(hypothesis_text)
        if int(expected_hypothesis_chars) != actual_hypothesis_chars:
            out["qe_hypothesis_chars_mismatch"] = {
                "expected": expected_hypothesis_chars,
                "actual": actual_hypothesis_chars,
            }
    for key, value in match.items():
        if key in {"candidate_text", "hypothesis_text", "source_text"}:
            continue
        out[key] = value
    return out
