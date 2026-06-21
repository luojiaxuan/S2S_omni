from s2s_omni.metrics import compression_ratio, lexical_recall, number_recall, token_count


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
