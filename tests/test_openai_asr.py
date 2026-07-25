from __future__ import annotations

import io
import urllib.error
from typing import Self

from s2s_omni import openai_asr


class FakeResponse:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return b'{"text":"ok"}'


def test_transcription_retries_transient_http_error(tmp_path, monkeypatch) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    attempts = 0

    def fake_urlopen(_request, timeout):
        nonlocal attempts
        assert timeout == 180.0
        attempts += 1
        if attempts == 1:
            raise urllib.error.HTTPError(
                "https://api.openai.com",
                500,
                "Internal Server Error",
                {},
                io.BytesIO(b""),
            )
        return FakeResponse()

    monkeypatch.setattr(openai_asr.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(openai_asr.time, "sleep", lambda _seconds: None)

    result = openai_asr.transcribe_openai_json(
        "key",
        "https://api.openai.com/v1",
        "gpt-4o-mini-transcribe",
        audio,
    )

    assert attempts == 2
    assert result == {"text": "ok"}
