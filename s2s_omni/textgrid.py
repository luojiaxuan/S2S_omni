from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SILENCE_LABELS = {"", "sil", "sp", "spn", "<eps>", "<sil>", "{sl}"}


@dataclass(frozen=True)
class TextGridInterval:
    tier: str
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class ChunkTimeSpan:
    start_s: float
    end_s: float
    unit_start: int
    unit_end: int


def parse_textgrid(path: str | Path) -> dict[str, list[TextGridInterval]]:
    """Parse common long-text Praat TextGrid files emitted by MFA."""

    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    tiers: dict[str, list[TextGridInterval]] = {}
    current_tier: str | None = None
    in_interval_tier = False
    pending_start: float | None = None
    pending_end: float | None = None
    in_item = False

    for raw in lines:
        line = raw.strip()
        if re.match(r"item \[\d+\]:", line):
            in_item = True
            current_tier = None
            in_interval_tier = False
            continue
        if not in_item:
            continue
        if line.startswith("class ="):
            in_interval_tier = _quoted_value(line) == "IntervalTier"
            continue
        if line.startswith("name ="):
            current_tier = _quoted_value(line)
            if in_interval_tier and current_tier:
                tiers.setdefault(current_tier, [])
            continue
        if not in_interval_tier or not current_tier:
            continue
        if line.startswith("xmin ="):
            pending_start = float(line.split("=", 1)[1].strip())
            continue
        if line.startswith("xmax ="):
            pending_end = float(line.split("=", 1)[1].strip())
            continue
        if line.startswith("text ="):
            text = _quoted_value(line)
            if pending_start is not None and pending_end is not None:
                tiers.setdefault(current_tier, []).append(
                    TextGridInterval(
                        tier=current_tier,
                        start_s=pending_start,
                        end_s=pending_end,
                        text=text,
                    )
                )
            pending_start = None
            pending_end = None
    return tiers


def choose_word_tier(tiers: dict[str, list[TextGridInterval]]) -> list[TextGridInterval]:
    if not tiers:
        raise ValueError("TextGrid has no interval tiers")
    preferred = ["words", "word", "characters", "chars", "phones"]
    by_lower = {name.lower(): name for name in tiers}
    for name in preferred:
        if name in by_lower and tiers[by_lower[name]]:
            return tiers[by_lower[name]]
    for name, intervals in tiers.items():
        if intervals and "phone" not in name.lower():
            return intervals
    return next(iter(tiers.values()))


def alignment_units(text: str) -> list[str]:
    units: list[str] = []
    for ch in text or "":
        if ch.isspace():
            continue
        category = unicodedata.category(ch)
        if category.startswith("P") or category.startswith("S"):
            continue
        units.append(ch)
    return units


def transcript_for_mfa(text: str) -> str:
    # Character-separated transcripts make chunk boundary mapping deterministic.
    # Punctuation is omitted because it has no acoustic interval.
    return " ".join(alignment_units(text))


def unit_prefix_counts(text: str) -> list[int]:
    counts = [0]
    total = 0
    for ch in text or "":
        if alignment_units(ch):
            total += 1
        counts.append(total)
    return counts


def nonsilence_intervals(intervals: Iterable[TextGridInterval]) -> list[TextGridInterval]:
    out = []
    for interval in intervals:
        label = normalize_label(interval.text)
        if label.lower() in SILENCE_LABELS:
            continue
        if interval.end_s <= interval.start_s:
            continue
        out.append(interval)
    return out


def chunk_time_spans_from_textgrid(
    textgrid_path: str | Path,
    full_text: str,
    char_spans: list[tuple[int, int]],
    *,
    min_duration_s: float = 0.02,
    max_unit_mismatch_ratio: float = 0.05,
) -> list[ChunkTimeSpan | None]:
    tiers = parse_textgrid(textgrid_path)
    intervals = nonsilence_intervals(choose_word_tier(tiers))
    expected_units = alignment_units(full_text)
    if not expected_units:
        raise ValueError("full_text has no MFA alignment units")
    mismatch = abs(len(intervals) - len(expected_units)) / max(1, len(expected_units))
    if mismatch > max_unit_mismatch_ratio:
        raise ValueError(
            f"MFA interval/unit mismatch: intervals={len(intervals)} units={len(expected_units)}"
        )
    usable_units = min(len(intervals), len(expected_units))
    prefix = unit_prefix_counts(full_text)

    spans: list[ChunkTimeSpan | None] = []
    for start_char, end_char in char_spans:
        unit_start = prefix[start_char]
        unit_end = prefix[end_char]
        if unit_start == unit_end:
            spans.append(None)
            continue
        if unit_start >= usable_units or unit_end > usable_units:
            raise ValueError(
                f"chunk unit span out of MFA range: {unit_start}:{unit_end} / {usable_units}"
            )
        start_s = intervals[unit_start].start_s
        end_s = intervals[unit_end - 1].end_s
        if end_s - start_s < min_duration_s:
            spans.append(None)
            continue
        spans.append(
            ChunkTimeSpan(
                start_s=float(start_s),
                end_s=float(end_s),
                unit_start=int(unit_start),
                unit_end=int(unit_end),
            )
        )
    return spans


def normalize_label(text: str) -> str:
    return str(text or "").strip().strip('"')


def _quoted_value(line: str) -> str:
    value = line.split("=", 1)[1].strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
    return value.replace('""', '"')

