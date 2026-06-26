from s2s_omni.floras_live import (
    coverage_status,
    parse_speeds,
    proportional_sentence_rows,
    split_sentences,
)


def test_parse_speeds_rejects_nonpositive():
    assert parse_speeds("1.0,1.5,2.0") == [1.0, 1.5, 2.0]
    try:
        parse_speeds("1.0,0")
    except ValueError:
        pass
    else:
        raise AssertionError("expected nonpositive speed to fail")


def test_sentence_rows_are_monotonic():
    rows = proportional_sentence_rows(["hello world", "next sentence"], 10.0, "en")
    assert len(rows) == 2
    assert rows[0]["source_start_s"] == 0.0
    assert rows[0]["source_end_s"] <= rows[1]["source_start_s"]
    assert rows[-1]["source_end_s"] == 10.0


def test_coverage_uses_reference_recall():
    row = coverage_status("the quick brown fox", "the quick brown fox jumps over other words")
    assert row["status"] == "covered"
    assert row["heuristic_recall"] == 1.0


def test_split_chinese_sentences():
    assert split_sentences("你好。今天很好！", "zh") == ["你好。", "今天很好！"]
