"""Text preprocessing for the Mental Health module (FR-08).

cleaning -> tokenisation -> stop-word removal (negations and first-person pronouns are KEPT,
because "not happy" and heavy "I/me" use are known distress signals) -> negation marking ->
Snowball stemming.
"""
from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

KEEP = {"no", "not", "nor", "never", "none", "nobody", "nothing", "neither", "nowhere", "cannot",
        "i", "me", "my", "myself", "alone", "against", "without", "nobody", "everything",
        "anything", "always", "less", "more", "most", "very", "too", "much"}
STOP_WORDS = frozenset(set(ENGLISH_STOP_WORDS) - KEEP)

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"[@#]\w+")
_SPECIAL_NEG = re.compile(r"\b(ca|wo)n['’]t\b")
_NEG = re.compile(r"n['’]t\b")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_CLAUSE = re.compile(r"[.,!?;:\n]+|\bbut\b|\bhowever\b")

NEG_PREFIX = "not_"
NEG_SCOPE = 2
PRONOUNS = frozenset({"i", "me", "my", "myself"})
NEGATORS = frozenset({"not", "no", "never", "nor", "cannot", "without", "nothing", "nobody",
                      "none", "neither", "nowhere", "hardly", "barely", "refused", "refuse",
                      "denied", "deny", "unable", "stopped", "skipped", "avoided", "lack"})
_stemmer = snowballstemmer.stemmer("english")


@lru_cache(maxsize=300_000)
def stem(word: str) -> str:
    return _stemmer.stemWord(word)


# ---- spelling normalisation of stretched words ("soooo", "dieee", "himm") -----------------
_RUN3 = re.compile(r"([a-z])\1{2,}")
_known: dict[str, int] = {}


def set_known_words(counts: dict[str, int]) -> None:
    """Vocabulary of real words (word -> frequency), built from the training corpus."""
    global _known
    _known = dict(counts)
    fix_elongation.cache_clear()


def build_known_words(texts, min_count: int = 5) -> dict[str, int]:
    from collections import Counter
    counts = Counter(w for t in texts for w in re.findall(r"[a-z]+", str(t).lower()))
    return {w: c for w, c in counts.items() if c >= min_count and not _RUN3.search(w)}


@lru_cache(maxsize=200_000)
def fix_elongation(w: str) -> str:
    """'emptyyy' -> 'empty', 'soooo' -> 'so', 'himm' -> 'him'; 'good', 'sleep', 'all' unchanged."""
    if _RUN3.search(w):
        one, two = _RUN3.sub(r"\1", w), _RUN3.sub(r"\1\1", w)
        if _known:
            return max((one, two), key=lambda c: _known.get(c, 0)) if (one in _known or two in _known) else two
        return two
    # trailing doubled letter that is not a real word: "himm" -> "him", "plzz" -> "plz"
    if _known and len(w) > 2 and w[-1] == w[-2] and w not in _known and w[:-1] in _known:
        return w[:-1]
    return w


def tokenize(text) -> list[str]:
    """Lower-case, clean and split the text; words in a negation scope get a ``not_`` prefix.

    Negation marking: after a negation word ("not", "never", "didn't", "refused", ...) the next
    NEG_SCOPE (2) content words of the same clause are prefixed, so "didn't attend the lectures"
    becomes ``not not_attend not_lectures``. The model then learns "not_happy" as a different
    feature from "happy" instead of reading both as the same positive word.
    """
    t = str(text).lower()
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _SPECIAL_NEG.sub(lambda m: "can not" if m.group(1) == "ca" else "will not", t)
    t = _NEG.sub(" not", t)
    tokens: list[str] = []
    for clause in _CLAUSE.split(t):
        words = [w for w in (fix_elongation(x) for x in _NON_ALPHA.sub(" ", clause).split())
                 if (len(w) > 1 or w == "i") and w not in STOP_WORDS]
        scope = 0
        for w in words:
            if w in NEGATORS:
                tokens.append("not" if w in ("cannot", "nor") else w)
                scope = NEG_SCOPE
            elif scope > 0 and w not in PRONOUNS:
                tokens.append(NEG_PREFIX + w)
                scope -= 1
            else:
                tokens.append(w)
    return tokens


def _stem_token(w: str) -> str:
    if w.startswith(NEG_PREFIX):
        return NEG_PREFIX + stem(w[len(NEG_PREFIX):])
    return stem(w)


def clean_text(text) -> str:
    return " ".join(_stem_token(w) for w in tokenize(text))


def clean_many(texts) -> list[str]:
    return [clean_text(t) for t in texts]


def stem_to_words(text) -> dict[str, str]:
    """Map each stem in the user's own text back to the word they wrote (for explanations)."""
    mapping: dict[str, str] = {}
    for w in tokenize(text):
        readable = ("not " + w[len(NEG_PREFIX):]) if w.startswith(NEG_PREFIX) else w
        mapping.setdefault(_stem_token(w), readable)
    return mapping
