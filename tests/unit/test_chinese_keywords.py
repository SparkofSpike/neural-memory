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
    extract_keywords,
    extract_weighted_keywords,
)

# Two-character words that a naive per-character stop-word filter destroys,
# because one of their characters is a standalone particle elsewhere:
# 会 (会议/社会/会计), 用 (用户), 更 (更新), 问 (问题), 果 (结果), 因 (原因).
COMMON_TWO_CHAR_WORDS = [
    "社会",
    "未来",
    "用户",
    "更新",
    "问题",
    "结果",
    "原因",
    "会议",
    "会计",
    "行为",
    "下载",
    "最近",
    "统一",
    "第一",
    "国家",
    "中国",
    "公司",
    "工作",
    "时间",
    "方法",
    "数据",
    "系统",
    "服务",
    "产品",
    "项目",
    "团队",
    "计划",
]


class TestCommonChineseWords:
    """A word must never be destroyed by a character inside it."""

    @pytest.mark.parametrize("word", COMMON_TWO_CHAR_WORDS)
    def test_two_char_word_survives(self, word: str) -> None:
        assert extract_keywords(word) == [word], (
            f"{word!r} lost its keyword — a character inside it was treated as a separator"
        )

    @pytest.mark.parametrize("word", ["媒体投稿", "神经记忆", "数据库", "服务器"])
    def test_four_char_word_survives(self, word: str) -> None:
        assert extract_keywords(word) == [word]


class TestChineseKeywordExtraction:
    """Short Chinese queries produce a usable keyword."""

    @pytest.mark.parametrize("text", ["媒体投稿", "记忆", "神经记忆", "覆巢"])
    def test_chinese_query_yields_keyword(self, text: str) -> None:
        assert extract_keywords(text) == [text]

    def test_single_han_character_dropped(self) -> None:
        """min_length=2 applies to Chinese too."""
        assert extract_keywords("猫") == []

    def test_pure_filler_yields_nothing(self) -> None:
        """A run with no content word should not invent one."""
        assert extract_keywords("的了是在") == []


class TestLongRunsAreNotGuessed:
    """A sentence is not a keyword, and guessing one out of it is worse than
    returning nothing: the wrong term retrieves unrelated memories and the
    fiber-summary tier answers from its top hit."""

    @pytest.mark.parametrize(
        "text",
        [
            "帮我看一下媒体投稿的记录",
            "我们明天开会讨论服务器迁移方案",
            "覆巢演出音效错位",
            "数据库连接超时",
        ],
    )
    def test_long_run_yields_no_keyword(self, text: str) -> None:
        assert extract_keywords(text) == [], (
            f"{text!r} produced a keyword by cutting a window out of a "
            f"sentence — that guess is what this guard exists to prevent"
        )

    def test_long_run_does_not_emit_a_verb(self) -> None:
        """The specific wrong answer found in review."""
        assert "讨论" not in extract_keywords("我们明天开会讨论服务器迁移方案")

    def test_long_run_does_not_emit_a_wide_prefix(self) -> None:
        """The specific wrong answer found in review."""
        assert "数据" not in extract_keywords("数据库连接超时")


class TestChineseEnglishMixed:
    """Mixed queries keep the Latin half, which already worked."""

    def test_english_half_preserved(self) -> None:
        keywords = extract_keywords("神经记忆 consolidation 失败")
        assert any("consolidation" in k for k in keywords)

    def test_identifier_weights_survive(self) -> None:
        weighted = extract_weighted_keywords("neural_memory 媒体投稿")
        texts = {kw.text for kw in weighted}
        assert "neural_memory" in texts, texts
        assert "媒体投稿" in texts, texts


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

    @pytest.mark.parametrize(
        "text",
        [
            "ReflexPipeline loader",
            "neural_memory.engine",
            "ValueError raised",
            "GEO_3.0 deployment",
            "getUserById processData",
        ],
    )
    def test_identifier_extraction_intact(self, text: str) -> None:
        """Code identifiers are the reason this function exists on the write
        path; the CJK branch must not disturb them."""
        assert extract_keywords(text)


class TestMinLength:
    """The caller's floor is honoured."""

    def test_min_length_filters_short_han_run(self) -> None:
        assert extract_keywords("记忆", min_length=3) == []

    def test_min_length_allows_longer_run(self) -> None:
        assert extract_keywords("媒体投稿", min_length=3) == ["媒体投稿"]

    def test_min_length_above_cap_yields_nothing(self) -> None:
        """A floor of 5 cannot be met without cutting a long run, and cutting
        is what this module refuses to do."""
        assert extract_keywords("服务器迁移", min_length=5) == []

    @pytest.mark.parametrize("text", ["服务器迁移", "数据库连接", "媒体投稿界面"])
    def test_no_keyword_shorter_than_min_length(self, text: str) -> None:
        for floor in (2, 3, 4, 5):
            short = [k for k in extract_keywords(text, min_length=floor) if len(k) < floor]
            assert not short, f"{text!r} floor={floor} produced {short}"


class TestChineseWeights:
    """Chinese keywords must carry sane, sortable weights."""

    def test_weights_are_positive(self) -> None:
        for kw in extract_weighted_keywords("媒体投稿 覆巢"):
            assert kw.weight > 0

    def test_results_sorted_descending(self) -> None:
        weighted = extract_weighted_keywords("媒体投稿 记忆 consolidation")
        weights = [kw.weight for kw in weighted]
        assert weights == sorted(weights, reverse=True)

    def test_no_duplicate_keywords(self) -> None:
        keywords = extract_keywords("媒体投稿 媒体投稿")
        assert len(keywords) == len(set(keywords))


class TestWritePath:
    """The same function runs while encoding, not only while recalling.

    A typical memory body is a full sentence, and a sentence is exactly what
    this module refuses to cut a keyword out of. So the write path gains no
    Chinese keyword — the same as before the change. What must hold is that it
    gains no *junk* keyword either: a sentence-length auto-tag would be worse
    than none.
    """

    @pytest.fixture()
    async def storage(self):
        from neural_memory.core.brain import Brain, BrainConfig
        from neural_memory.storage.memory_store import InMemoryStorage

        store = InMemoryStorage()
        brain = Brain.create(name="test_zh_write", config=BrainConfig())
        await store.save_brain(brain)
        store.set_brain(brain.id)
        return store

    @pytest.mark.asyncio()
    async def test_sentence_body_writes_no_junk_tag(self, storage) -> None:
        """A refused keyword must not become a sentence-length tag."""
        from neural_memory.core.brain import BrainConfig
        from neural_memory.engine.encoder import MemoryEncoder

        encoder = MemoryEncoder(storage, BrainConfig())
        content = "帮我看一下媒体投稿的记录并把结果发给我"
        result = await encoder.encode(content=content)

        for tag in result.fiber.tags:
            assert len(tag) <= 4, f"sentence-length tag written: {tag!r}"

    @pytest.mark.asyncio()
    async def test_short_body_may_carry_a_keyword(self, storage) -> None:
        """A body that is itself a short term is a legitimate keyword."""
        from neural_memory.core.brain import BrainConfig
        from neural_memory.engine.encoder import MemoryEncoder

        encoder = MemoryEncoder(storage, BrainConfig())
        result = await encoder.encode(content="媒体投稿")

        raw = [
            n.metadata["_raw_keywords"]
            for n in result.neurons_created
            if n.metadata.get("_raw_keywords")
        ]
        assert raw, "short Chinese body produced no _raw_keywords"
        assert any("媒体投稿" in group for group in raw), raw
