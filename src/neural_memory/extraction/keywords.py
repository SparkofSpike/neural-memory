"""Keyword extraction from text with Vietnamese word segmentation support."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from neural_memory.utils.cjk import is_cjk_char

logger = logging.getLogger(__name__)

# English stop words — standard
STOP_WORDS_EN: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "and",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "what",
        "which",
        "who",
        "whom",
    }
)

# Conversational English — common contractions without apostrophes,
# filler words, and profanity that keyword extraction should skip.
# These produce noisy bigrams ("like dont", "fucking fall") when
# left in the token stream.
_STOP_WORDS_CONVERSATIONAL_EN: frozenset[str] = frozenset(
    {
        # Contractions without apostrophes (casual typing)
        "dont",
        "doesnt",
        "didnt",
        "wont",
        "wouldnt",
        "couldnt",
        "shouldnt",
        "cant",
        "isnt",
        "arent",
        "wasnt",
        "werent",
        "hasnt",
        "havent",
        "hadnt",
        "im",
        "ive",
        "id",
        "youre",
        "youve",
        "youll",
        "youd",
        "hes",
        "shes",
        "its",
        "were",
        "theyre",
        "theyve",
        "theyll",
        "thats",
        "theres",
        "heres",
        "whats",
        "lets",
        # Common filler / hedging
        "like",
        "really",
        "actually",
        "basically",
        "literally",
        "honestly",
        "seriously",
        "obviously",
        "suppose",
        "guess",
        "thing",
        "things",
        "something",
        "anything",
        "everything",
        "nothing",
        "kinda",
        "sorta",
        "gonna",
        "gotta",
        "wanna",
        "dunno",
        "yeah",
        "nah",
        "yep",
        "nope",
        "hey",
        "oh",
        "ugh",
        "lol",
        "lmao",
        "idk",
        "tbh",
        "imo",
        "imho",
        # Profanity (no topical signal)
        "fucking",
        "fuck",
        "shit",
        "damn",
        "hell",
        "crap",
        # Common verbs that rarely carry topical signal
        "think",
        "know",
        "want",
        "make",
        "go",
        "going",
        "come",
        "take",
        "give",
        "tell",
        "say",
        "said",
        "get",
        "got",
        "went",
        "put",
        "look",
        "looking",
    }
)

# Vietnamese stop words
STOP_WORDS_VI: frozenset[str] = frozenset(
    {
        "và",
        "của",
        "là",
        "có",
        "được",
        "cho",
        "với",
        "này",
        "trong",
        "để",
        "các",
        "những",
        "một",
        "đã",
        "tôi",
        "bạn",
        "anh",
        "chị",
        "em",
        "ở",
        "tại",
        "khi",
        "thì",
        "mà",
        "nếu",
        "vì",
        "cũng",
        "như",
        "từ",
        "đến",
        "lại",
        "ra",
        "vào",
        "lên",
        "xuống",
        "rồi",
        "sẽ",
        "đang",
        "vẫn",
        "còn",
        "chỉ",
        "rất",
        "quá",
        "làm",
        "gì",
        "sao",
        "nào",
        "đâu",
        "ai",
        "bao",
        "nhiêu",
    }
)

# Combined stop words for backward compatibility
STOP_WORDS: frozenset[str] = STOP_WORDS_EN | STOP_WORDS_VI | _STOP_WORDS_CONVERSATIONAL_EN

# Han characters present anywhere in the text. Used as a cheap pre-check so
# Latin-only input never pays for a per-character scan.
_CJK_ANY = re.compile(
    "[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    "\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f"
    "\U0002b820-\U0002ceaf\U0002ceb0-\U0002ebef\U00030000-\U0003134f]"
)

# Longest Han run still treated as one term. A run has no internal word
# boundaries, so a window cut out of it would be a guess, and a guess that lands
# on a common but unrelated word retrieves unrelated memories --
# `_build_fts_query` phrases the term as a single FTS phrase and the
# fiber-summary tier answers from its top hit. So the whole run is used or
# nothing is.
#
# Five rather than four because a run of five is still a term, not a sentence:
# 服务器迁移 and 角色管理页 are ordinary domain words. It is a boundary, not a
# solution: a six-character term still yields nothing, because an exact phrase
# that long stops appearing in stored summaries and any window cut out of one
# would be a guess. Segmentation is what would actually lift this bound.
_CJK_MAX_WHOLE_RUN = 5

# Characters with no retrieval signal on their own. Deliberately **not** used
# to split a run: a character being filler on its own says nothing about its
# role inside a word -- 会 belongs to 社会/会议/会计, 用 to 用户, 更 to 更新,
# 问 to 问题. Used by the density rule below, and by the all-filler test.
_CJK_FILLER_CHARS: frozenset[str] = frozenset(
    "的了是在和与就都而及或吧呢吗啊嗯哦我你他她它这那有看把被让给个们"
)

# Pronouns, question words, modals, and the verbs of a chat request. A run made
# mostly of these is an utterance, not a term -- 帮我看一下, 我想知道, 怎么样,
# 是不是, 然后呢. Unlike `_CJK_FILLER_CHARS` this is not a claim that the
# character is meaningless; it only feeds a ratio.
_CJK_CHAT_CHARS: frozenset[str] = frozenset(
    ""
    "我你他她它咱您"
    "帮请让求给"
    "想要知道觉得能否会可应该需"
    "怎什么哪怎样多少几"
    "么啊吧呢吗呀哦嗯"
    "不没别勿"
    "这那样种些"
    "就还也都又再才"
    "来去说做看查找问答"
    "一下个点"
    "非常很太挺更最"
    "如果但所以因然后话"
    "已现正将曾经刚"
    ""
)

# A run at least this long in which at least this share of the characters are
# chat/filler characters is treated as an utterance rather than a term. Ratios
# are not applied to two-character runs: 统一 and 第一 are content words whose
# second character happens to be common, and a two-character sample cannot
# support a ratio.
_CJK_UTTERANCE_MIN_LEN = 3
_CJK_UTTERANCE_RATIO = 0.4

# Whole phrases that carry no retrieval signal. These cannot be caught by a
# character test without also destroying content words -- 一 would take
# 第一/统一/唯一 with it, 已 would take 自己 -- and 一个 slipped through the
# character rule precisely because 一 is absent from the set while 个 is
# present. The list is a backstop for the ratio rule, not a general solution.
_CJK_FILLER_WORDS: frozenset[str] = frozenset(
    {
        "什么",
        "怎么",
        "怎样",
        "怎么样",
        "怎么办",
        "为什么",
        "如果",
        "但是",
        "所以",
        "因为",
        "一个",
        "一下",
        "我们",
        "你们",
        "他们",
        "这个",
        "那个",
        "好的",
        "可以",
        "已经",
        "现在",
        "就是",
        "还是",
        "或者",
        "而且",
        "然后",
        "多少",
        "几个",
        "一些",
        "非常",
        "应该",
        "需要",
        "帮我",
    }
)

# Conjunctions that join two topics in Chinese. Used as a run separator **only**
# when both sides keep at least two characters, so 用户和权限 yields 用户 and
# 权限 while 和平 / 和谐 / 总和 stay intact -- 和 is a content character when it
# starts or ends the run.
_CJK_CONJUNCTIONS = frozenset("和与及或")

# Zero-width characters survive copy-paste from web pages and split a term in
# half without being whitespace or Han. 媒体\u200b投稿 -> 媒体 without this.
_CJK_ZERO_WIDTH = str.maketrans(dict.fromkeys("\u200b\u200c\u200d\ufeff"))


def _split_conjunctions(run: str) -> list[str]:
    """Split a run on conjunctions that sit between two content fragments."""
    parts: list[str] = []
    start = 0
    for i, ch in enumerate(run):
        if ch in _CJK_CONJUNCTIONS and i - start >= 2 and len(run) - i - 1 >= 2:
            parts.append(run[start:i])
            start = i + 1
    parts.append(run[start:])
    return [p for p in parts if p]


def _is_cjk_utterance(run: str) -> bool:
    """True when a run is mostly pronouns, question words and modal verbs."""
    if len(run) < _CJK_UTTERANCE_MIN_LEN:
        return False
    chat = sum(1 for ch in run if ch in _CJK_CHAT_CHARS or ch in _CJK_FILLER_CHARS)
    return chat / len(run) >= _CJK_UTTERANCE_RATIO


def _cjk_runs(text: str) -> list[str]:
    """Return every run of consecutive Han characters, in reading order.

    Zero-width characters are dropped first so a term pasted from a web page
    is not split in half. Runs are then split on inter-fragment conjunctions.
    """
    runs: list[str] = []
    current: list[str] = []
    for ch in text.translate(_CJK_ZERO_WIDTH):
        if is_cjk_char(ch):
            current.append(ch)
        else:
            if current:
                runs.extend(_split_conjunctions("".join(current)))
            current = []
    if current:
        runs.extend(_split_conjunctions("".join(current)))
    return runs


def _extract_cjk_keyword(text: str, min_length: int) -> str | None:
    """Pick a Chinese keyword for the spaced FTS index, or None.

    Every run is validated first, then the longest survivor is returned. Both
    halves of that order matter: a run that is too long must not suppress a
    shorter usable one (``媒体投稿 服务器迁移方案`` must still yield 媒体投稿),
    and an all-filler run must not win on length (``我们的 数据`` must yield
    数据). When two survivors are the same length the first in reading order
    wins, so the result is deterministic.

    Refusing to cut a long run is deliberate. A two- or five-character window
    of a sentence is as likely to be a verb or a particle as the topic
    (``我们明天开会讨论服务器迁移方案`` -> ``讨论``;
    ``数据库连接超时`` -> ``数据``), and a wrong term silently retrieves the
    wrong memories. Returning None for an over-long run leaves the caller where
    it was before this path existed, which is honest; segmentation is what
    would actually solve it.

    Exactly one keyword is returned because ``_build_fts_query`` joins keywords
    with an implicit AND, so emitting several competing runs from one query
    narrows the result set to text containing all of them.
    """
    if not _CJK_ANY.search(text):
        return None
    usable = [
        run
        for run in _cjk_runs(text)
        if min_length <= len(run) <= _CJK_MAX_WHOLE_RUN
        and run not in _CJK_FILLER_WORDS
        and not all(ch in _CJK_FILLER_CHARS for ch in run)
        and not _is_cjk_utterance(run)
    ]
    if not usable:
        return None
    return max(usable, key=len)


# Vietnamese diacritical character pattern (unique to Vietnamese, not French)
_VI_DIACRITICS = re.compile(r"[ăâđêôơưắằẳẵặấầẩẫậếềểễệốồổỗộớờởỡợứừửữự]")


def _detect_vietnamese(text: str) -> bool:
    """Detect if text contains Vietnamese based on diacritical characters."""
    return bool(_VI_DIACRITICS.search(text.lower()))


def _get_stop_words(language: str, text: str) -> frozenset[str]:
    """Get appropriate stop words for the detected language."""
    if language == "vi":
        return STOP_WORDS_VI
    if language == "en":
        return STOP_WORDS_EN | _STOP_WORDS_CONVERSATIONAL_EN
    # "auto" — use all
    return STOP_WORDS


_PYVI_WARNED = False


def _tokenize_vietnamese(text: str) -> str | None:
    """Try to tokenize Vietnamese text using pyvi.

    Returns tokenized text with compound words joined by underscores,
    or None if pyvi is not available.
    """
    global _PYVI_WARNED
    try:
        import warnings

        with warnings.catch_warnings():
            # pyvi emits NumPy VisibleDeprecationWarning (UserWarning subclass, NOT
            # DeprecationWarning) during pickle.load at module import. Narrow
            # category filters miss it — suppress everything in this scoped block.
            warnings.simplefilter("ignore")
            from pyvi import ViTokenizer

        return ViTokenizer.tokenize(text)  # type: ignore[no-any-return]
    except ImportError:
        if not _PYVI_WARNED:
            logger.warning(
                "Vietnamese text detected but pyvi is not installed — "
                "keyword extraction will produce low-quality bigrams. "
                "Install with: pip install pyvi"
            )
            _PYVI_WARNED = True
        return None


@dataclass(frozen=True)
class WeightedKeyword:
    """A keyword with an importance weight.

    Attributes:
        text: The keyword text (unigram or bi-gram)
        weight: Importance weight (0.0 - 1.5), higher = more important
    """

    text: str
    weight: float


def extract_weighted_keywords(
    text: str,
    min_length: int = 2,
    language: str = "auto",
) -> list[WeightedKeyword]:
    """
    Extract weighted keywords with bi-gram support.

    Scoring factors:
    - Position: earlier words score higher (1.0 → 0.5 linear decay)
    - Bi-grams: adjacent non-stop-word pairs get averaged weight * 1.2 boost

    For Vietnamese text, uses pyvi word segmentation to detect compound words
    (e.g., "học sinh" → single keyword "học sinh" instead of separate "học", "sinh").

    Args:
        text: The text to extract from
        min_length: Minimum word length for unigrams
        language: Language hint ("vi", "en", or "auto")

    Returns:
        List of WeightedKeyword sorted by weight descending
    """
    is_vietnamese = language == "vi" or (language == "auto" and _detect_vietnamese(text))
    stop_words = _get_stop_words(language, text)

    # For Vietnamese: try pyvi tokenization to detect compound words
    tokenized_text = text
    if is_vietnamese:
        vi_tokenized = _tokenize_vietnamese(text)
        if vi_tokenized is not None:
            tokenized_text = vi_tokenized

    # ── Phase 1: Extract code identifiers BEFORE lowercasing ─────────
    # These are preserved as-is (case-sensitive) for precise code recall.
    code_identifiers: dict[str, float] = {}
    code_spans: list[tuple[int, int]] = []  # spans to skip in natural-language pass

    # PascalCase: ReflexPipeline, MemoryEncoder (also split into sub-words)
    for m in re.finditer(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", tokenized_text):
        ident = m.group(1)
        code_identifiers[ident] = 1.2  # high weight for compound identifiers
        code_spans.append((m.start(), m.end()))
        # Also add split sub-words: ReflexPipeline -> reflex, pipeline
        parts = re.findall(r"[A-Z][a-z]+", ident)
        for part in parts:
            code_identifiers.setdefault(part.lower(), 0.7)

    # camelCase: processData, getKeywords
    for m in re.finditer(r"\b([a-z][a-z0-9]*(?:[A-Z][a-z0-9]+)+)\b", tokenized_text):
        ident = m.group(1)
        code_identifiers[ident] = 1.2
        code_spans.append((m.start(), m.end()))
        parts = re.findall(r"[a-z][a-z0-9]*|[A-Z][a-z0-9]+", ident)
        for part in parts:
            code_identifiers.setdefault(part.lower(), 0.7)

    # snake_case (2+ segments): extract_keywords, activate_trail
    for m in re.finditer(r"\b([a-z][a-z0-9]*(?:_[a-z][a-z0-9]*){1,})\b", tokenized_text):
        ident = m.group(1)
        code_identifiers[ident] = 1.2
        code_spans.append((m.start(), m.end()))
        for part in ident.split("_"):
            if len(part) >= min_length:
                code_identifiers.setdefault(part, 0.7)

    # Dotted modules: neural_memory.engine, os.path
    for m in re.finditer(r"\b([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){1,})\b", tokenized_text):
        code_identifiers[m.group(1)] = 1.2
        code_spans.append((m.start(), m.end()))

    # Error types: ValueError, ConnectionRefusedError
    for m in re.finditer(r"\b([A-Z][a-zA-Z]*(?:Error|Exception))\b", tokenized_text):
        code_identifiers[m.group(1)] = 1.1
        code_spans.append((m.start(), m.end()))

    # ── Phase 2: Natural language keywords (existing logic) ────────
    # Han characters are absent from the Latin range below, so a Chinese query
    # tokenizes to nothing and recall answers "no relevant memories" for text
    # that is indexed and retrievable. Extract the Chinese term separately.
    cjk_keyword = _extract_cjk_keyword(text, min_length)

    words = re.findall(r"\b[a-zA-ZÀ-ỹ]+(?:_[a-zA-ZÀ-ỹ]+)*\b", tokenized_text.lower())

    # Filter to content words with original position
    filtered: list[tuple[str, int]] = [
        (w, i)
        for i, w in enumerate(words)
        if len(w.replace("_", "")) >= min_length
        and w.replace("_", " ") not in stop_words
        and w not in stop_words
    ]

    if not filtered and not code_identifiers and not cjk_keyword:
        return []

    total = max(len(filtered), 1)
    weighted: dict[str, float] = {}

    # Seed with code identifiers (high priority)
    weighted.update(code_identifiers)

    # The Chinese term is what the query is about; rank it with the leading
    # natural-language words.
    if cjk_keyword is not None:
        weighted[cjk_keyword] = max(weighted.get(cjk_keyword, 0.0), 1.0)

    # Unigrams with position decay (1.0 at start -> 0.5 at end)
    for idx, (word, _orig_pos) in enumerate(filtered):
        position_weight = 1.0 - 0.5 * (idx / max(1, total - 1))
        # Store with underscores replaced by spaces for readability
        display_word = word.replace("_", " ")
        weighted[display_word] = max(weighted.get(display_word, 0.0), position_weight)

    # Bi-grams from adjacent non-stop words within 3 original word positions
    for i in range(len(filtered) - 1):
        w1, p1 = filtered[i]
        w2, p2 = filtered[i + 1]
        if p2 - p1 <= 3:
            dw1 = w1.replace("_", " ")
            dw2 = w2.replace("_", " ")
            bigram = f"{dw1} {dw2}"
            bigram_weight = (weighted.get(dw1, 0.5) + weighted.get(dw2, 0.5)) / 2 * 1.2
            weighted[bigram] = max(weighted.get(bigram, 0.0), bigram_weight)

    results = [WeightedKeyword(text=k, weight=v) for k, v in weighted.items()]
    results.sort(key=lambda x: x.weight, reverse=True)
    return results


def extract_keywords(
    text: str,
    min_length: int = 2,
    language: str = "auto",
) -> list[str]:
    """
    Extract keywords from text, sorted by importance.

    Backward-compatible wrapper around extract_weighted_keywords().
    Returns bi-grams before unigrams, ordered by weight.

    Args:
        text: The text to extract from
        min_length: Minimum word length
        language: Language hint ("vi", "en", or "auto")

    Returns:
        List of keyword strings
    """
    weighted = extract_weighted_keywords(text, min_length, language=language)
    return [kw.text for kw in weighted]
