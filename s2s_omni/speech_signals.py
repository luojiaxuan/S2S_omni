from __future__ import annotations

from collections import Counter
from typing import Any

from .metrics import unit_count
from .schema import S2SSample
from .style import style_violations


def duration_signals(
    sample: S2SSample,
    target_text: str,
    speech_duration_s: float | None,
) -> dict[str, Any]:
    target_units = unit_count(target_text, sample.tgt_lang)
    budget_s = sample.target.max_target_duration_s or sample.metadata.get("allowed_speech_s")
    reference_tts_s = sample.metadata.get("reference_tts_duration_s")
    source_wall_s = sample.metadata.get("source_wall_duration_s") or sample.timing.src_duration_s
    default_rate = sample.metadata.get("default_target_unit_rate")

    out: dict[str, Any] = {
        "target_units": target_units,
        "speech_duration_s": _round(speech_duration_s),
        "budget_duration_s": _round(_float_or_none(budget_s)),
        "reference_tts_duration_s": _round(_float_or_none(reference_tts_s)),
        "source_wall_duration_s": _round(_float_or_none(source_wall_s)),
        "style_violations": style_violations(target_text, sample.tgt_lang),
    }
    if speech_duration_s and speech_duration_s > 0:
        out["target_units_per_s"] = _round(target_units / speech_duration_s)
        out["target_units_per_min"] = _round(target_units / speech_duration_s * 60.0)
        budget_f = _float_or_none(budget_s)
        if budget_f and budget_f > 0:
            out["duration_budget_ratio"] = _round(speech_duration_s / budget_f)
        reference_f = _float_or_none(reference_tts_s)
        if reference_f and reference_f > 0:
            out["duration_reference_tts_ratio"] = _round(speech_duration_s / reference_f)
        source_f = _float_or_none(source_wall_s)
        if source_f and source_f > 0:
            out["speech_s2s_rtf"] = _round(speech_duration_s / source_f)
        rate_f = _float_or_none(default_rate)
        if rate_f and rate_f > 0:
            estimated_default_s = target_units / rate_f
            out["estimated_default_text_duration_s"] = _round(estimated_default_s)
            out["rate_pressure_ratio"] = _round(estimated_default_s / speech_duration_s)
    return out


def coverage_signals(
    target_text: str,
    asr_text: str | None,
    lang: str,
) -> dict[str, Any]:
    target_units = text_units(target_text, lang)
    asr_units = text_units(asr_text or "", lang)
    lcs_len = _lcs_len(target_units, asr_units)
    greedy = greedy_ordered_match(target_units, asr_units)
    bag_overlap = _bag_overlap(target_units, asr_units)
    target_n = len(target_units)
    asr_n = len(asr_units)
    return {
        "asr_text": asr_text or "",
        "target_coverage_units": lcs_len,
        "target_coverage_recall": _safe_ratio(lcs_len, target_n),
        "asr_precision_vs_target": _safe_ratio(lcs_len, asr_n),
        "ordered_target_recall": _safe_ratio(greedy["matched_units"], target_n),
        "last_matched_target_unit_ratio": greedy["last_matched_target_unit_ratio"],
        "unmatched_target_suffix_units": greedy["unmatched_target_suffix_units"],
        "bag_target_recall": _safe_ratio(bag_overlap, target_n),
        "bag_asr_precision": _safe_ratio(bag_overlap, asr_n),
    }


def gate_reasons(
    duration: dict[str, Any],
    coverage: dict[str, Any] | None = None,
    max_duration_budget_ratio: float = 1.05,
    min_coverage_recall: float = 0.9,
    min_last_matched_ratio: float = 0.9,
    max_rate_pressure_ratio: float = 1.35,
) -> list[str]:
    reasons: list[str] = []
    duration_ratio = duration.get("duration_budget_ratio")
    if duration_ratio is not None and duration_ratio > max_duration_budget_ratio:
        reasons.append("speech_over_budget")
    rate_pressure = duration.get("rate_pressure_ratio")
    if rate_pressure is not None and rate_pressure > max_rate_pressure_ratio:
        reasons.append("speech_too_fast_for_text")
    for reason in duration.get("style_violations") or []:
        reasons.append(f"text_style:{reason}")
    if coverage is not None:
        recall = coverage.get("target_coverage_recall")
        if recall is not None and recall < min_coverage_recall:
            reasons.append("low_asr_target_coverage")
        last_ratio = coverage.get("last_matched_target_unit_ratio")
        if last_ratio is not None and last_ratio < min_last_matched_ratio:
            reasons.append("early_stop_or_suffix_missing")
    return reasons


def text_units(text: str | None, lang: str) -> list[str]:
    text = normalize_for_coverage(text or "", lang)
    if lang.startswith(("zh", "ja", "ko")):
        return [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
    return text.split()


def normalize_for_coverage(text: str, lang: str) -> str:
    text = (text or "").lower()
    if lang.startswith(("zh", "ja", "ko")):
        return "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff")
    out = []
    last_space = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            last_space = False
        elif not last_space:
            out.append(" ")
            last_space = True
    return "".join(out).strip()


def greedy_ordered_match(target_units: list[str], asr_units: list[str]) -> dict[str, Any]:
    target_index = 0
    last_matched = -1
    for unit in asr_units:
        while target_index < len(target_units) and target_units[target_index] != unit:
            target_index += 1
        if target_index >= len(target_units):
            break
        last_matched = target_index
        target_index += 1
    matched = last_matched + 1 if last_matched >= 0 else 0
    return {
        "matched_units": matched,
        "last_matched_target_unit_ratio": _safe_ratio(last_matched + 1, len(target_units)),
        "unmatched_target_suffix_units": max(0, len(target_units) - matched),
    }


def _lcs_len(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    if len(right) > len(left):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_unit in left:
        current = [0]
        for idx, right_unit in enumerate(right, start=1):
            if left_unit == right_unit:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1]


def _bag_overlap(left: list[str], right: list[str]) -> int:
    return sum((Counter(left) & Counter(right)).values())


def _safe_ratio(value: float, total: float) -> float | None:
    if total <= 0:
        return None
    return _round(value / total)


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)
