from __future__ import annotations

import importlib.util
import json
import sys
import wave
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


target_speech = load_script(
    "acl6060_target_speech_instances",
    ROOT / "scripts/build_acl6060_target_speech_instances.py",
)


def test_transcribe_windows_reuses_kit_gpt_mini_text(tmp_path: Path, monkeypatch) -> None:
    audio_path = tmp_path / "target.wav"
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)
    (tmp_path / "target_asr_windows.jsonl").write_text(
        json.dumps(
            {
                "window_index": 0,
                "start_s": 0.0,
                "end_s": 1.0,
                "asr_model": "gpt-4o-mini-transcribe",
                "asr_text": "复用文本。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_transcribe(_api_key, _base_url, model, _window_path, **_kwargs):
        assert model == "whisper-1"
        return {
            "text": "复用文本",
            "words": [{"word": "复用文本", "start": 0.1, "end": 0.9}],
        }

    monkeypatch.setattr(target_speech, "transcribe_openai_json", fake_transcribe)
    rows = target_speech.transcribe_windows(
        audio_path=audio_path,
        sample_dir=tmp_path,
        api_key="test",
        base_url="https://example.test",
        asr_model="gpt-4o-mini-transcribe",
        timestamp_model="whisper-1",
        target_lang="zh",
        window_s=120.0,
        resume=True,
    )
    assert rows[0]["asr_text"] == "复用文本。"
    assert rows[0]["asr_reused_from"] == "target_asr_windows.jsonl"


def test_align_hypothesis_units_interpolates_punctuation() -> None:
    timed = [
        {
            "unit": "你",
            "normalized": "你",
            "audio_start_ms": 100.0,
            "audio_end_ms": 200.0,
        },
        {
            "unit": "好",
            "normalized": "好",
            "audio_start_ms": 200.0,
            "audio_end_ms": 300.0,
        },
    ]
    ends, summary = target_speech.align_hypothesis_units("你，好", "zh", timed)
    assert ends == [200.0, 250.0, 300.0]
    assert summary["alignment_coverage"] == 1.0


def test_align_hypothesis_units_normalizes_traditional_chinese() -> None:
    timed = target_speech.timed_alignment_units(
        [
            {
                "word": "學習",
                "start_s": 0.1,
                "end_s": 0.3,
            }
        ],
        "zh",
    )
    ends, summary = target_speech.align_hypothesis_units("学习", "zh", timed)
    assert ends == [200.0, 300.0]
    assert summary["alignment_coverage"] == 1.0


def test_unit_playout_uses_audio_position_and_packet_queue() -> None:
    packets = target_speech.packet_playout_timeline(
        [
            {
                "received_at_ms": 100.0,
                "sent_source_ms": 500.0,
                "duration_ms": 200.0,
                "audio_start_ms": 0.0,
                "audio_end_ms": 200.0,
            },
            {
                "received_at_ms": 150.0,
                "sent_source_ms": 900.0,
                "duration_ms": 200.0,
                "audio_start_ms": 200.0,
                "audio_end_ms": 400.0,
            },
        ]
    )
    delays, elapsed = target_speech.unit_playout_times(
        [50.0, 250.0, 400.0],
        packets,
        [(100.0, 500.0), (300.0, 900.0)],
        1000.0,
    )
    assert delays == [500.0, 900.0, 900.0]
    assert elapsed == [150.0, 350.0, 500.0]


def test_kit_source_timeline_uses_completed_post_time(tmp_path: Path) -> None:
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "postStats": [
                    {"sent_at_s": 1.2, "audio_end_s": 0.96},
                    {"sent_at_s": 2.7, "audio_end_s": 1.92},
                ]
            }
        ),
        encoding="utf-8",
    )
    timeline = target_speech.source_send_timeline(tmp_path, "kit", 1920.0)
    assert timeline == [(1200.0, 960.0), (2700.0, 1920.0)]
    assert target_speech.source_consumed_at_playout(1199.0, timeline) == 0.0
    assert target_speech.source_consumed_at_playout(1200.0, timeline) == 960.0
    assert target_speech.source_consumed_at_playout(3000.0, timeline) == 1920.0
