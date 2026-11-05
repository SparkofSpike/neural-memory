"""Tests for Chinese (CJK) keyword extraction.

Chinese text has no spaces, so the Latin-oriented tokenizer in
``extract_weighted_keywords`` dropped every Han character: a query like
``媒体投稿`` produced ``[]``, which left ``Stimulus.keywords`` empty and made
recall answer "No relevant memories found" for text that was indexed and
retrievable. These tests pin the Chinese path so it cannot silently regress.
"""

from __future__ import annotations

import pytest

from neural_memory.extraction.keywords import (
    CJK_STOP_CHARS,
    extract_keywords,
    extract_weighted_keywords,
)


class TestChineseKeywordExtraction:
    """Chinese queries must yield at least one usable keyword."""

    @pytest.mark.parametrize(
        "text",
        [
            "媒体投稿",
            "记忆",
            "神经记忆",
            "覆巢演出音效错位",
            "数据库连接超时",
        ],
    )
    def test_chinese_query_yields_keyword(self, text: str) -> None:
        keywords = extract_keywords(text)
        assert keywords, f"Chinese query {text!r} produced no keywords"

    def test_short_query_kept_whole(self) -> None:
        """A run of four or fewer Han characters is a plausible word."""
        assert "媒体投稿" in extract_keywords("媒体投稿")

    def test_long_run_is_not_emitted_whole(self) -> None:
        """A long Han run is a sentence, not a word.

        Emitting it verbatim would build an FTS phrase that matches nothing.
        """
        keywords = extract_keywords("帮我看一下媒体投稿的记录")
        assert "帮我看一下媒体投稿的记录" not in keywords

    def test_long_run_recovers_head_words(self) -> None:
        """The long run must still yield something that matches the index."""
        keywords = extract_keywords("帮我看一下媒体投稿的记录")
        assert any(k.startswith("媒体") for k in keywords), keywords

    def test_single_han_character_dropped(self) -> None:
        """min_length=2 applies to Chinese too."""
        keywords = extract_keywords("猫")
        assert "猫" not in keywords

    def test_stop_characters_filtered(self) -> None:
        """Function words must not become keywords on their own."""
        keywords = extract_keywords("的了是在和与就都而")
        assert not keywords, keywords

    def test_greeting_only_query_yields_nothing(self) -> None:
        """Pure filler should not produce a false-positive keyword."""
        keywords = extract_keywords("帮我看一下")
        assert not keywords, keywords

    def test_min_length_is_honored_for_long_runs(self) -> None:
        """The capped head must not shrink below the caller's floor."""
        for text in ("服务器迁移情况", "数据库连接超时问题"):
            for floor in (2, 3, 5):
                keywords = extract_keywords(text, min_length=floor)
                short = [k for k in keywords if len(k) < floor]
                assert not short, f"{text!r} floor={floor} produced {short}"

    def test_stop_char_set_is_non_empty(self) -> None:
        assert CJK_STOP_CHARS
        assert "的" in CJK_STOP_CHARS


class TestChineseEnglishMixed:
    """Mixed queries keep both halves."""

    def test_english_half_preserved(self) -> None:
        keywords = extract_keywords("神经记忆 consolidation 失败")
        assert any("consolidation" in k for k in keywords)

    def test_chinese_half_preserved(self) -> None:
        keywords = extract_keywords("神经记忆 consolidation 失败")
        assert any(k == "神经记忆" for k in keywords), keywords

    def test_identifier_weights_survive(self) -> None:
        weighted = extract_weighted_keywords("neural_memory 媒体投稿")
        texts = {kw.text for kw in weighted}
        assert "neural_memory" in texts, texts
        assert any("媒体" in t for t in texts), texts


class TestNoChineseRegression:
    """Text without Han characters must behave exactly as before."""

    @pytest.mark.parametrize(
        "text",
        [
            "The weather is nice today",
            "neural memory bug",
            "Hôm nay trời đẹp",
        ],
    )
    def test_non_chinese_unchanged(self, text: str) -> None:
        keywords = extract_keywords(text)
        assert keywords
        for k in keywords:
            assert not any("\u4e00" <= ch <= "\u9fff" for ch in k)

    def test_empty_input(self) -> None:
        assert extract_keywords("") == []

    def test_whitespace_only(self) -> None:
        assert extract_keywords("   ") == []

    def test_english_weights_still_ordered(self) -> None:
        weighted = extract_weighted_keywords("python programming language")
        assert weighted
        weights = [kw.weight for kw in weighted]
        assert weights == sorted(weights, reverse=True)


class TestChineseWeights:
    """Chinese keywords must carry sane, sortable weights."""

    def test_weights_are_positive(self) -> None:
        for kw in extract_weighted_keywords("媒体投稿 覆巢演出音效错位"):
            assert kw.weight > 0

    def test_results_sorted_descending(self) -> None:
        weighted = extract_weighted_keywords("媒体投稿 记忆 consolidation")
        weights = [kw.weight for kw in weighted]
        assert weights == sorted(weights, reverse=True)

    def test_no_duplicate_keywords(self) -> None:
        keywords = extract_keywords("媒体投稿 媒体投稿")
        assert len(keywords) == len(set(keywords))
