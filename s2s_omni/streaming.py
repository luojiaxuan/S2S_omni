from __future__ import annotations

from dataclasses import replace

from .metrics import token_count
from .schema import CompressionTarget, S2SSample


def estimate_duration_s(text: str, source_wpm: float = 155.0) -> float:
    words = max(1, token_count(text))
    return words / source_wpm * 60.0


def choose_mode(target_ratio: float) -> str:
    if target_ratio >= 0.85:
        return "faithful"
    if target_ratio >= 0.65:
        return "concise"
    return "very_concise"


def make_stress_sample(
    sample: S2SSample,
    speed_factor: float,
    max_end_lag_s: float,
    preferred_target_wpm: float,
) -> S2SSample:
    scaled_timing = sample.timing.scaled_source(speed_factor)
    src_duration = scaled_timing.src_duration_s
    if src_duration is None:
        src_duration = estimate_duration_s(sample.source_text) / speed_factor

    max_target_duration_s = src_duration + max_end_lag_s
    reference = sample.reference_translation or sample.compressed_translation or ""
    reference_duration = estimate_duration_s(reference, source_wpm=preferred_target_wpm)
    target_ratio = min(1.0, max_target_duration_s / max(reference_duration, 0.1))
    max_target_words = max(4, int(preferred_target_wpm * max_target_duration_s / 60.0))

    target = CompressionTarget(
        mode=choose_mode(target_ratio),
        target_ratio=round(target_ratio, 3),
        max_target_words=max_target_words,
        max_target_duration_s=round(max_target_duration_s, 3),
        max_target_wpm=preferred_target_wpm,
        max_end_lag_s=max_end_lag_s,
        preserve=sample.target.preserve,
        compressible=sample.target.compressible,
    )

    metadata = dict(sample.metadata)
    metadata.update(
        {
            "source_speed_factor": speed_factor,
            "base_id": sample.id,
            "stress_kind": "source_speedup",
        }
    )
    return replace(
        sample,
        id=f"{sample.id}__speed_{speed_factor:g}",
        timing=scaled_timing,
        target=target,
        metadata=metadata,
    )


def expand_speed_conditions(
    samples: list[S2SSample],
    speed_factors: list[float],
    max_end_lag_s: float,
    preferred_target_wpm: float,
) -> list[S2SSample]:
    expanded: list[S2SSample] = []
    for sample in samples:
        for speed_factor in speed_factors:
            expanded.append(
                make_stress_sample(
                    sample,
                    speed_factor=speed_factor,
                    max_end_lag_s=max_end_lag_s,
                    preferred_target_wpm=preferred_target_wpm,
                )
            )
    return expanded
