from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from s2s_omni.hibiki_zero import (
    boundaries_from_textgrid,
    chunk_word_spans,
    english_mfa_transcript,
    english_words,
    gate_duration_rtf,
    normalize_src_lang,
    speech_s2s_rtf_for_chunks,
)


def write_textgrid(path: Path, labels: list[str], step: float = 0.2) -> None:
    intervals = []
    for idx, label in enumerate(labels, start=1):
        xmin = (idx - 1) * step
        xmax = idx * step
        intervals.append(
            textwrap.dedent(
                f"""
                intervals [{idx}]:
                    xmin = {xmin}
                    xmax = {xmax}
                    text = "{label}"
                """
            ).strip()
        )
    body = "\n        ".join(intervals)
    path.write_text(
        textwrap.dedent(
            f"""
            File type = "ooTextFile"
            Object class = "TextGrid"

            xmin = 0
            xmax = {len(labels) * step}
            tiers? <exists>
            size = 1
            item []:
                item [1]:
                    class = "IntervalTier"
                    name = "words"
                    xmin = 0
                    xmax = {len(labels) * step}
                    intervals: size = {len(labels)}
                    {body}
            """
        ).strip(),
        encoding="utf-8",
    )


def test_language_scope() -> None:
    assert normalize_src_lang("French") == "fr"
    assert normalize_src_lang("de-DE") == "de"
    with pytest.raises(ValueError):
        normalize_src_lang("it")


def test_english_words_and_mfa_transcript() -> None:
    assert english_words("It's 10:30, hello-world!") == ["it's", "10", "30", "hello", "world"]
    assert english_mfa_transcript("Hello, WORLD!") == "hello world"


def test_chunk_word_spans() -> None:
    assert chunk_word_spans(["hello world", "", "this is fine"]) == [(0, 2), (2, 2), (2, 5)]


def test_boundaries_from_textgrid(tmp_path: Path) -> None:
    textgrid = tmp_path / "sample.TextGrid"
    write_textgrid(textgrid, ["hello", "world", "this", "is", "fine"])
    spans = boundaries_from_textgrid(
        textgrid,
        "hello world this is fine",
        ["hello world", "", "this is fine"],
    )
    assert spans[0] is not None
    assert spans[0].start_s == 0.0
    assert spans[0].end_s == 0.4
    assert spans[1] is None
    assert spans[2] is not None
    assert spans[2].start_s == 0.4
    assert spans[2].end_s == 1.0


def test_rtf_gate() -> None:
    rtf = speech_s2s_rtf_for_chunks([0.5, 1.5, None], [1.0, 1.0, 2.0])
    assert rtf == [0.5, 1.5, None]
    gate = gate_duration_rtf(rtf, threshold=1.0)
    assert gate["pass"] is False
    assert gate["violation_rate"] == 0.5
