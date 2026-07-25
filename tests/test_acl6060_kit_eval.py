from __future__ import annotations

import argparse
import importlib.util
import json
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


kit_eval = load_script("acl6060_kit_eval", ROOT / "scripts/run_acl6060_kit_live_eval.py")
kit_stream = load_script("acl6060_kit_stream", ROOT / "scripts/run_kit_live_floras.py")
comparison = load_script(
    "acl6060_kit_quality_mode_comparison",
    ROOT / "scripts/build_acl6060_kit_quality_mode_comparison.py",
)


def test_kit_audio_post_records_source_send_wall_time(monkeypatch) -> None:
    ticks = iter([10.2, 10.7])
    monkeypatch.setattr(kit_stream.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        kit_stream,
        "request_json_or_text",
        lambda **_kwargs: (200, True, "OK"),
    )
    result = kit_stream.post_audio_chunk(
        base_url="https://example.test",
        session_id="session",
        cookie_header="cookie",
        pcm_chunk=b"\x00\x00" * 160,
        start_s=0.0,
        run_start_monotonic=10.0,
    )
    assert result["request_started_at_s"] == 0.2
    assert result["sent_at_s"] == 0.7
    assert result["ms"] == 500


def test_session_name_includes_quality_mode() -> None:
    args = argparse.Namespace(
        target_lang="zh",
        format="mixed",
        tts_quality_mode="low_latency",
        chunk_ms=960,
        speed_factor=1.0,
    )
    assert kit_eval.session_name(args, 2, "talk") == (
        "acl6060_enzh_kit_mixed_low_latency_chunk960_speed1_002_talk"
    )


def test_session_name_uses_default_for_empty_sanitized_mode() -> None:
    args = argparse.Namespace(
        target_lang="de",
        format="online",
        tts_quality_mode="---",
        chunk_ms=1920,
        speed_factor=1.5,
    )
    assert kit_eval.session_name(args, 0, "talk") == (
        "acl6060_ende_kit_online_default_chunk1920_speed1p5_000_talk"
    )


def test_upsert_jsonl_replaces_duplicate_index(tmp_path: Path) -> None:
    path = tmp_path / "responses.jsonl"
    kit_eval.upsert_jsonl(path, {"index": 1, "value": "old"})
    kit_eval.upsert_jsonl(path, {"index": 0, "value": "zero"})
    kit_eval.upsert_jsonl(path, {"index": 1, "value": "new"})
    assert kit_eval.read_jsonl(path) == [
        {"index": 0, "value": "zero"},
        {"index": 1, "value": "new"},
    ]


def test_comparison_config_diff_and_audio_groups() -> None:
    high = {"chunk_ms": 960, "kit_tts_quality_mode": "high_quality"}
    low = {"chunk_ms": 960, "kit_tts_quality_mode": "low_latency"}
    assert comparison.config_diff(high, low) == {
        "kit_tts_quality_mode": ["high_quality", "low_latency"]
    }
    assert comparison.consecutive_audio_groups(
        [{"wav": "a.wav"}, {"wav": "a.wav"}, {"wav": "b.wav"}]
    ) == [("a.wav", [0, 1]), ("b.wav", [2])]


def test_canonical_baseline_validation(tmp_path: Path) -> None:
    row = {
        "Language": "En-Zh",
        "Speedup": "1x",
        "System": "KIT Lecture Translator",
        "BLEU": "35.4007",
        "XCOMET-XL": "0.680956",
        "LongYAAL": "85358.7596",
        "Ending Offset": "166172.1296",
    }
    table = tmp_path / "table.jsonl"
    table.write_text(json.dumps(row) + "\n", encoding="utf-8")
    selected = comparison.canonical_baseline_row(table)
    comparison.validate_canonical_baseline(
        selected,
        {
            "BLEU": 35.4007,
            "XCOMET-XL": 0.6809561740252175,
            "LongYAAL": 85358.7596,
            "Ending Offset": 166172.12965,
        },
    )


def test_ending_offset_reconciliation() -> None:
    rows = [
        {"mode": "high_quality", "Ending Offset": 10.0},
        {"mode": "high_quality", "Ending Offset": 20.0},
        {"mode": "low_latency", "Ending Offset": 5.0},
        {"mode": "low_latency", "Ending Offset": 15.0},
    ]
    assert comparison.validate_ending_offset_reconciliation(
        rows,
        {"Ending Offset": 15.0},
        {"Ending Offset": 10.0},
    ) == {
        "high_quality_per_talk_mean": 15.0,
        "high_quality_aggregate": 15.0,
        "low_latency_per_talk_mean": 10.0,
        "low_latency_aggregate": 10.0,
    }
