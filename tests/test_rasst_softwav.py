from __future__ import annotations

from s2s_omni.rasst import atempo_filters, parse_rasst_row
from s2s_omni.textgrid import chunk_time_spans_from_textgrid, transcript_for_mfa


def test_parse_plain_rasst_row_chunks() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "<audio>"},
            {"role": "assistant", "content": "你好，"},
            {"role": "user", "content": "<audio>"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "<audio>"},
            {"role": "assistant", "content": "世界"},
        ],
        "audios": [
            "/x/AUD0001/42/0.wav",
            "/x/AUD0001/42/1.wav",
            "/x/AUD0001/42/2.wav",
        ],
    }
    parsed = parse_rasst_row(row, "train.jsonl", 3)
    assert parsed.row_id == "AUD0001_42"
    assert parsed.full_target_text == "你好，世界"
    assert [turn.assistant_text for turn in parsed.turns] == ["你好，", "", "世界"]
    assert parsed.target_char_spans == [(0, 3), (3, 3), (3, 5)]


def test_transcript_for_mfa_removes_punctuation() -> None:
    assert transcript_for_mfa("你好，世界。") == "你 好 世 界"


def test_chunk_time_spans_from_character_textgrid(tmp_path) -> None:
    tg = tmp_path / "x.TextGrid"
    tg.write_text(
        """
File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 4
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 4
        intervals: size = 4
        intervals [1]:
            xmin = 0.0
            xmax = 0.5
            text = "你"
        intervals [2]:
            xmin = 0.5
            xmax = 1.0
            text = "好"
        intervals [3]:
            xmin = 1.0
            xmax = 1.5
            text = "世"
        intervals [4]:
            xmin = 1.5
            xmax = 2.0
            text = "界"
""",
        encoding="utf-8",
    )
    spans = chunk_time_spans_from_textgrid(tg, "你好，世界", [(0, 3), (3, 5)])
    assert spans[0] is not None
    assert spans[1] is not None
    assert spans[0].start_s == 0.0
    assert spans[0].end_s == 1.0
    assert spans[1].start_s == 1.0
    assert spans[1].end_s == 2.0


def test_atempo_filter_chains_large_speed() -> None:
    assert atempo_filters(1.7) == "atempo=1.7"
    assert atempo_filters(4.0) == "atempo=2,atempo=2"
