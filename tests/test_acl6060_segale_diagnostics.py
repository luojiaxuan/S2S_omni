from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


diagnostics = load_script(
    "acl6060_segale_diagnostics",
    ROOT / "scripts/build_acl6060_segale_diagnostics.py",
)


def test_speech_playout_fields_use_sentence_end_as_the_origin() -> None:
    result = diagnostics.latency_fields(
        {"source_length": 3000, "elapsed": [2500, 4500, 7000], "raw_units": ["a", "b"]}
    )
    assert result == {
        "first_speech_playout_offset_ms": -500.0,
        "ending_offset_ms": 4000.0,
        "speech_playout_span_ms": 4500.0,
        "target_units": 2,
    }


def test_null_sentinel_has_no_latency() -> None:
    result = diagnostics.latency_fields({"source_length": None, "elapsed": [], "raw_units": []})
    assert result["first_speech_playout_offset_ms"] is None
    assert result["ending_offset_ms"] is None
    assert result["speech_playout_span_ms"] is None
    assert result["target_units"] == 0


def test_target_audio_timing_fields_use_last_spoken_playout() -> None:
    result = diagnostics.target_audio_timing_fields(
        [
            {
                "index": 0,
                "timing_method": diagnostics.TARGET_SPEECH_TIMING_METHOD,
                "target_audio_last_arrival_ms": 6500,
                "target_audio_playout_end_ms": 8000,
                "target_speech_last_unit_playout_ms": 7200,
            }
        ],
        [{"index": 0, "source_length": 6000}],
    )
    assert result == {
        "talk_speech_final_offset_mean_ms": 1200.0,
        "target_audio_queue_tail_mean_ms": 1500.0,
        "target_audio_after_speech_mean_ms": 800.0,
    }


def test_structural_alignment_label_does_not_claim_semantic_match() -> None:
    assert diagnostics.structural_alignment_label({"null_alignment_type": ""}) == (
        "non-null SEGALE group"
    )
    assert (
        diagnostics.structural_alignment_label({"null_alignment_type": "under_translation"})
        == "null under_translation"
    )


def test_paired_delta_summary_excludes_null_alignments() -> None:
    def row(xcomet: float, ending: float, first: float, null: str = ""):
        return {
            "xcomet_xl_score": xcomet,
            "ending_offset_ms": ending,
            "first_speech_playout_offset_ms": first,
            "null_alignment_type": null,
        }

    by_speed = {
        1.0: {1: row(0.5, 1000, 200), 2: row(0.8, 1500, 300)},
        1.25: {1: row(0.7, 900, 150), 2: row(0.0, 0, 0, "under_translation")},
        1.5: {1: row(0.6, 1200, 400), 2: row(0.9, 1400, 250)},
    }
    result = diagnostics.paired_delta_row("En-Zh", "GPT", "openai", "zh", by_speed)
    assert result["source_sentences_all_speeds"] == 2
    assert result["valid_pairs_1p25x_vs_1x"] == 1
    assert abs(result["xcomet_delta_mean_1p25x_vs_1x"] - 0.2) < 1e-9
    assert result["ending_offset_delta_p50_ms_1p25x_vs_1x"] == -100.0
    assert result["valid_pairs_1p5x_vs_1x"] == 2
    assert result["ending_offset_delta_p50_ms_1p5x_vs_1x"] == 50.0
