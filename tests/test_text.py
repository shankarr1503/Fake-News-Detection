from common.text import clean_text, combine_title_text


def test_clean_text_splits_on_punctuation_instead_of_gluing_words():
    assert clean_text("A well-known end.Next") == "known end"


def test_clean_text_removes_digits_unicode_punctuation_and_stopwords():
    assert clean_text("“Breaking” — 2016 election: the RESULTS!") == "breaking election results"


def test_clean_text_handles_non_strings():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_combine_title_text():
    assert combine_title_text("Title", "Body") == "Title Body"
    assert combine_title_text(float("nan"), "Body") == "Body"
