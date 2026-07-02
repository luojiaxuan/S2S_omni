from s2s_omni.metrics import (
    compression_ratio,
    heuristic_score_sample,
    lexical_recall,
    number_recall,
    s2s_real_time_factor,
    token_count,
)
from s2s_omni.floras_live import corpus_metrics
from s2s_omni.schema import S2SSample


def test_token_count_counts_words_and_cjk_chars():
    assert token_count("hello world 123") == 3
    assert token_count("hello 世界") == 3


def test_compression_ratio():
    ratio = compression_ratio("a b", "a b c d")
    assert ratio == 0.5


def test_lexical_recall():
    assert lexical_recall(["supplier audit", "June 28"], "Supplier audit moved to June 28") == 1.0


def test_number_recall():
    assert number_recall("ship on 28 and 30", "28 only") == 0.5


def test_s2s_real_time_factor_from_metadata():
    sample = S2SSample(
        id="x__speed_2",
        src_lang="en",
        tgt_lang="zh",
        source_text="hello",
        reference_translation="你好世界",
        metadata={"default_target_unit_rate": 4.0, "playback_budget_s": 1.0},
    )
    assert s2s_real_time_factor("你好世界", sample) == 1.0


def test_heuristic_score_reports_s2s_rtf_violation():
    sample = S2SSample(
        id="x__speed_2",
        src_lang="en",
        tgt_lang="zh",
        source_text="hello",
        reference_translation="你好世界啊",
        metadata={
            "default_target_unit_rate": 4.0,
            "playback_budget_s": 1.0,
            "rtf_threshold": 1.0,
        },
    )
    row = heuristic_score_sample(sample, "你好世界啊")
    assert row["s2s_rtf"] == 1.25
    assert row["s2s_rtf_violation"] is True


def test_corpus_metrics_uses_chinese_bleu_tokenizer():
    row = corpus_metrics("你好世界今天很好", "你好世界今天不错", "zh")
    assert row["bleu_tokenizer"] == "zh"
    assert row["bleu"] > 0
