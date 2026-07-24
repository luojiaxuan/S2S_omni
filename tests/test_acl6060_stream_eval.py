from __future__ import annotations

import base64
import importlib.util
import sys
import wave
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "projects"
    / "acl6060_s2s_metrics_seed"
    / "run_acl6060_live_stream_eval.py"
)
SPEC = importlib.util.spec_from_file_location("acl6060_stream_eval", SCRIPT)
assert SPEC is not None
acl6060_stream_eval = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = acl6060_stream_eval
SPEC.loader.exec_module(acl6060_stream_eval)


def test_resolve_hf_release_paths_and_audio_rows(tmp_path: Path) -> None:
    input_dir = tmp_path / "main_result" / "inputs" / "acl_zh"
    audio_dir = tmp_path / "main_result" / "audio" / "acl6060"
    input_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    wav_path = audio_dir / "2022.acl-long.268.wav"
    wav_path.write_bytes(b"placeholder")
    (input_dir / "source.list").write_text(
        "data/main_result/audio/acl6060/2022.acl-long.268.wav\n",
        encoding="utf-8",
    )
    (input_dir / "target.list").write_text("目标全文\n", encoding="utf-8")
    (input_dir / "ref.txt").write_text("目标句子\n", encoding="utf-8")
    (input_dir / "source_text.txt").write_text("source sentence\n", encoding="utf-8")
    (input_dir / "audio.yaml").write_text(
        "- wav: data/main_result/audio/acl6060/2022.acl-long.268.wav\n"
        "  offset: 0.0\n"
        "  duration: 1.5\n",
        encoding="utf-8",
    )

    paths = acl6060_stream_eval.resolve_paths(tmp_path, "zh")
    assert paths.source_list == input_dir / "source.list"
    rows = acl6060_stream_eval.parse_simple_audio_yaml(paths.audio_yaml)
    assert rows == [
        {
            "wav": "data/main_result/audio/acl6060/2022.acl-long.268.wav",
            "offset": 0.0,
            "duration": 1.5,
        }
    ]
    assert (
        acl6060_stream_eval.resolve_audio_path(
            "data/main_result/audio/acl6060/2022.acl-long.268.wav",
            tmp_path,
            paths,
        )
        == wav_path
    )


def test_zh_text_delta_uses_non_whitespace_units() -> None:
    streamed = acl6060_stream_eval.StreamedText([], [], [], [], [])
    acl6060_stream_eval.add_text_delta(
        streamed,
        "你 好。",
        "zh",
        delay_ms=960.0,
        elapsed_ms=1000.0,
    )
    assert "".join(streamed.text_parts) == "你 好。"
    assert streamed.delays_ms == [960.0, 960.0, 960.0]
    assert streamed.elapsed_ms == [1000.0, 1000.0, 1000.0]


def test_pcm_segment_ranges_respects_session_limit() -> None:
    pcm = b"\x00\x00" * 10
    assert acl6060_stream_eval.pcm_segment_ranges(pcm, 10, 0) == [(0, 20)]
    assert acl6060_stream_eval.pcm_segment_ranges(pcm, 10, 0.4) == [
        (0, 8),
        (8, 16),
        (16, 20),
    ]


def test_atempo_filters_handles_ffmpeg_bounds() -> None:
    assert acl6060_stream_eval.atempo_filters(1.5) == "atempo=1.50000000"
    assert acl6060_stream_eval.atempo_filters(4.0) == "atempo=2.0,atempo=2.00000000"
    assert acl6060_stream_eval.atempo_filters(0.25) == "atempo=0.5,atempo=0.50000000"


def test_output_audio_extractors_and_wav_writer(tmp_path: Path) -> None:
    pcm = b"\x01\x00\x02\x00"
    encoded = base64.b64encode(pcm).decode("ascii")
    event = {
        "serverContent": {
            "modelTurn": {
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "audio/pcm;rate=24000",
                            "data": encoded,
                        }
                    }
                ]
            }
        }
    }
    assert acl6060_stream_eval.gemini_output_audio(event) == [(encoded, 24000)]
    assert acl6060_stream_eval.gemini_output_audio(
        {
            "serverContent": {
                "modelTurn": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/pcm",
                                "data": encoded,
                            }
                        },
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": encoded,
                            }
                        },
                    ]
                }
            }
        }
    ) == [(encoded, 24000)]
    assert acl6060_stream_eval.openai_output_audio(
        {"type": "session.output_audio.delta", "delta": encoded}
    ) == (encoded, 24000)
    assert acl6060_stream_eval.openai_output_audio(
        {
            "type": "session.output_audio.delta",
            "delta": encoded,
            "sample_rate": 16000,
        }
    ) == (encoded, 16000)

    streamed = acl6060_stream_eval.StreamedText([], [], [], [], [])
    acl6060_stream_eval.append_base64_pcm_audio(streamed, encoded, 24000)
    output = tmp_path / "target.wav"
    assert acl6060_stream_eval.write_streamed_audio(streamed, output) == output
    with wave.open(str(output), "rb") as handle:
        assert handle.getframerate() == 24000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.readframes(handle.getnframes()) == pcm


def test_audio_capture_error_does_not_raise() -> None:
    streamed = acl6060_stream_eval.StreamedText([], [], [], [], [])
    error = acl6060_stream_eval.capture_base64_pcm_audio(
        streamed,
        "not-base64",
        24000,
        "openai",
    )
    assert error is not None
    assert error["provider"] == "openai"
    assert "audio_capture_error" in error
    assert streamed.audio_parts == []

    encoded = base64.b64encode(b"\x01\x00").decode("ascii")
    assert (
        acl6060_stream_eval.capture_base64_pcm_audio(
            streamed,
            encoded,
            24000,
            "openai",
        )
        is None
    )
    rate_error = acl6060_stream_eval.capture_base64_pcm_audio(
        streamed,
        encoded,
        16000,
        "openai",
    )
    assert rate_error is not None
    assert "sample rate changed" in rate_error["audio_capture_error"]
