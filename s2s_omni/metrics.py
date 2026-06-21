from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any

from .schema import S2SSample

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[\u4e00-\u9fff]")
NUMBER_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def token_count(text: str) -> int:
    return len(tokenize(text))


def cjk_char_count(text: str) -> int:
    return sum(1 for ch in text or "" if "\u4e00" <= ch <= "\u9fff")


def unit_count(text: str, lang: str | None = None) -> int:
    """Count target-side budget units: CJK chars for zh/ja/ko, tokens otherwise."""

    lang = (lang or "").lower()
    if lang.startswith(("zh", "ja", "ko")):
        cjk_count = cjk_char_count(text)
        if cjk_count:
            return cjk_count
    return token_count(text)


def compression_ratio(candidate: str, reference: str | None) -> float | None:
    if not reference:
        return None
    ref_n = max(1, token_count(reference))
    return token_count(candidate) / ref_n


def target_unit_ratio(candidate: str, reference: str | None, lang: str) -> float | None:
    if not reference:
        return None
    ref_n = max(1, unit_count(reference, lang))
    return unit_count(candidate, lang) / ref_n


def budget_units(sample: S2SSample) -> int | None:
    if sample.target.max_target_chars is not None:
        return sample.target.max_target_chars
    return sample.target.max_target_words


def budget_ratio(candidate: str, sample: S2SSample) -> float | None:
    budget = budget_units(sample)
    if budget is None or budget <= 0:
        return None
    return unit_count(candidate, sample.tgt_lang) / budget


def lexical_recall(required_items: list[str], candidate: str) -> float | None:
    if not required_items:
        return None
    candidate_norm = normalize(candidate)
    hits = 0
    for item in required_items:
        if normalize(item) in candidate_norm:
            hits += 1
    return hits / len(required_items)


def number_recall(source_or_reference: str, candidate: str) -> float | None:
    required = list(dict.fromkeys(NUMBER_RE.findall(source_or_reference or "")))
    if not required:
        return None
    candidate_numbers = Counter(NUMBER_RE.findall(candidate or ""))
    hits = 0
    for number in required:
        if candidate_numbers[number] > 0:
            hits += 1
            candidate_numbers[number] -= 1
    return hits / len(required)


def words_per_minute(text: str, duration_s: float | None) -> float | None:
    if duration_s is None or duration_s <= 0:
        return None
    return token_count(text) / (duration_s / 60.0)


def units_per_minute(text: str, lang: str, duration_s: float | None) -> float | None:
    if duration_s is None or duration_s <= 0:
        return None
    return unit_count(text, lang) / (duration_s / 60.0)


def end_lag_s(sample: S2SSample) -> float | None:
    if sample.timing.src_end_s is None or sample.timing.tgt_end_s is None:
        return None
    return sample.timing.tgt_end_s - sample.timing.src_end_s


def listenability_penalty(wpm: float | None, max_wpm: float | None) -> float:
    if wpm is None or max_wpm is None or max_wpm <= 0:
        return 0.0
    return max(0.0, (wpm - max_wpm) / max_wpm)


def budget_penalty(ratio: float | None) -> float:
    if ratio is None:
        return 0.0
    return max(0.0, ratio - 1.0)


def lag_penalty(lag_s: float | None, max_end_lag_s: float | None) -> float:
    if lag_s is None or max_end_lag_s is None or max_end_lag_s <= 0:
        return 0.0
    return max(0.0, (lag_s - max_end_lag_s) / max_end_lag_s)


def bag_f1(reference: str | None, candidate: str) -> float | None:
    if not reference:
        return None
    ref = Counter(tokenize(reference))
    cand = Counter(tokenize(candidate))
    if not ref or not cand:
        return 0.0
    overlap = sum((ref & cand).values())
    precision = overlap / sum(cand.values())
    recall = overlap / sum(ref.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def optional_sacrebleu(candidates: list[str], references: list[str]) -> dict[str, Any]:
    """Return corpus BLEU/chrF when sacrebleu is installed, otherwise a skip note."""

    if not candidates or not references:
        return {"available": False, "reason": "empty_inputs"}
    try:
        import sacrebleu  # type: ignore
    except ModuleNotFoundError:
        return {"available": False, "reason": "sacrebleu_not_installed"}
    bleu = sacrebleu.corpus_bleu(candidates, [references])
    chrf = sacrebleu.corpus_chrf(candidates, [references])
    result = {
        "available": True,
        "bleu": round(float(bleu.score), 6),
        "chrf": round(float(chrf.score), 6),
    }
    signatures = {
        "bleu": getattr(bleu, "signature", None),
        "chrf": getattr(chrf, "signature", None),
    }
    signatures = {key: str(value) for key, value in signatures.items() if value}
    if signatures:
        result["signature"] = signatures
    return result


def heuristic_score_sample(sample: S2SSample, candidate: str | None = None) -> dict[str, Any]:
    candidate = candidate or sample.candidate_translation or sample.preferred_target_text()
    if not candidate:
        raise ValueError(f"sample {sample.id} has no candidate text")

    reference = sample.reference_translation or sample.compressed_translation
    lexical_ratio = compression_ratio(candidate, reference)
    unit_ratio = target_unit_ratio(candidate, reference, sample.tgt_lang)
    budget = budget_units(sample)
    over_budget_ratio = budget_ratio(candidate, sample)
    term_recall = lexical_recall(sample.must_keep_terms, candidate)
    number_source = reference or sample.source_text
    num_recall = number_recall(number_source, candidate)
    f1 = bag_f1(reference, candidate)

    duration_budget = sample.target.max_target_duration_s or sample.timing.tgt_duration_s
    wpm = words_per_minute(candidate, duration_budget)
    upm = units_per_minute(candidate, sample.tgt_lang, duration_budget)
    lag = end_lag_s(sample)

    speed_pen = listenability_penalty(wpm, sample.target.max_target_wpm)
    budget_pen = budget_penalty(over_budget_ratio)
    late_pen = lag_penalty(lag, sample.target.max_end_lag_s)
    term_score = 1.0 if term_recall is None else term_recall
    number_score = 1.0 if num_recall is None else num_recall
    f1_score = 1.0 if f1 is None else f1
    budget_score = 1.0 - min(1.0, budget_pen)

    # A compact scalar for quick sweeps. LLM judge remains the main semantic metric.
    aggregate = (
        0.3 * f1_score
        + 0.25 * term_score
        + 0.2 * number_score
        + 0.05 * budget_score
        + 0.1 * (1.0 - min(1.0, speed_pen))
        + 0.1 * (1.0 - min(1.0, late_pen))
    )

    return {
        "id": sample.id,
        "candidate_tokens": token_count(candidate),
        "candidate_units": unit_count(candidate, sample.tgt_lang),
        "reference_tokens": token_count(reference or ""),
        "reference_units": unit_count(reference or "", sample.tgt_lang),
        "budget_units": budget,
        "compression_ratio": lexical_ratio,
        "target_unit_reference_ratio": unit_ratio,
        "target_budget_ratio": over_budget_ratio,
        "target_budget_violation": (
            over_budget_ratio is not None and over_budget_ratio > 1.0
        ),
        "bag_f1_vs_reference": f1,
        "must_keep_term_recall": term_recall,
        "number_recall": num_recall,
        "estimated_target_wpm": wpm,
        "estimated_target_units_per_minute": upm,
        "end_lag_s": lag,
        "budget_penalty": budget_pen,
        "listenability_penalty": speed_pen,
        "lag_penalty": late_pen,
        "heuristic_aggregate": round(float(aggregate), 6),
    }


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"count": len(rows)}
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    for key in numeric_keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        summary[key] = {
            "count": len(values),
            "mean": round(statistics.fmean(values), 6),
            "p50": percentile(values, 0.5),
            "p90": percentile(values, 0.9),
            "p95": percentile(values, 0.95),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
        }
    bool_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, bool)
        }
    )
    for key in bool_keys:
        summary[f"{key}_rate"] = rate(rows, key)
    return summary


def rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if key in row]
    if not values:
        return None
    return round(sum(1 for value in values if value) / len(values), 6)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    values = sorted(float(v) for v in values)
    if len(values) == 1:
        return round(values[0], 6)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    if lo == hi:
        return round(values[lo], 6)
    frac = pos - lo
    return round(values[lo] * (1.0 - frac) + values[hi] * frac, 6)


def normalize(text: str) -> str:
    return " ".join(tokenize(text))
