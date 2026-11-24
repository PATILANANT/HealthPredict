"""Rule-based safety layer for the Mental Health module.

The ML model only knows the 7 categories of its training data. None of them covers threats of
violence towards other people, and words like "kill" also appear in harmless slang
("this traffic is killing me"). Critical phrases are therefore checked with explicit rules that
run alongside the model. The rules can only RAISE the risk level, never lower it.

Matching works on a "squashed" copy of the text: lower-case, letters only, every run of a
repeated letter collapsed to one ("killl himmm" -> "kil him", "soooo" -> "so"), so stretched
or doubled spellings still match. Patterns are squashed the same way.
"""
from __future__ import annotations

import re

SELF_HARM_MSG = ("If you are having thoughts of self-harm, please reach out now: Tele-MANAS 14416 "
                 "or 1-800-891-4416 (India, 24x7, free), or your local emergency number.")
HARM_OTHERS_MSG = ("Your text mentions hurting someone. Strong anger after being hurt is common, but "
                   "if you feel you might act on it, please step away and talk to someone today: "
                   "Tele-MANAS 14416 (24x7, free) or a counsellor. If anyone is in immediate danger, "
                   "call 112.")

_VIOLENT = r"(kill|murder|stab|shoot|hurt|beat|attack|destroy|finish|slap|punch|harm|burn)"
_TARGET = (r"(him|her|them|you|that (guy|man|girl|woman|person|bitch|bastard)|this (guy|man)|"
           r"my (ex|boss|father|dad|mother|mom|brother|sister|husband|wife|boyfriend|girlfriend)|"
           r"everyone|everybody|someone|somebody)")
_INTENT = r"(want to|wanna|going to|gonna|will|wil|shall|am going to|would|plan to|planning to|i ll|ill)"

SELF_HARM = [
    r"(kill|hurt|harm|cut|hang|shoot) (my ?self|me)",
    r"(end|take|ending|taking) my (own )?life",
    r"suicid",
    r"(want|wanna|wana|going|gonna|ready|wish i could|i wish i could) (to )?die",
    r"i wish i (was|were) dead",
    r"(i ll|il|i wil|gonna) end it (al|tonight|today)",
    r"(do not|dont) want to (live|be alive|wake up|exist)",
    r"better of (dead|without me)",
    r"no reason to live",
    r"(jump|jumping) (of|from) (the |a )?(bridge|building|roof|terrace)",
    r"overdose",
]

HARM_OTHERS = [
    rf"{_INTENT} {_VIOLENT} {_TARGET}",
    rf"{_VIOLENT} (him|her|them) (for|because|if)",
    r"(wont|wil not|not going to|never) spare (him|her|them|that|you|this)",
    r"(make|made) (him|her|them) pay",
    r"(take|get|taking) (my )?revenge",
    r"(he|she|they) (wil|is going to|deserve|deserves|wil deserve|diserve|diserves)\b.{0,15}\b(worst|die|death|pay|pain|suffer)",
    r"(want|wish) (him|her|them) (dead|to die|to suffer)",
]


def squash(text: str) -> str:
    t = str(text).lower().replace("’", "'")
    t = re.sub(r"\b(ca|wo)n't\b", lambda m: "can not" if m.group(1) == "ca" else "wont", t)
    t = re.sub(r"n't\b", " not", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"([a-z])\1+", r"\1", t)          # collapse repeated letters
    return " " + re.sub(r"\s+", " ", t).strip() + " "


def _compile(patterns):
    out = []
    for p in patterns:
        # squash the literal letters in the pattern the same way as the text
        p = re.sub(r"([a-z])\1+", r"\1", p)
        out.append(re.compile(r"(?<![a-z])" + p + r"(?![a-z])"))
    return out


_SELF = _compile(SELF_HARM)
_OTHERS = _compile(HARM_OTHERS)


def _find(rules, text):
    hits = []
    for rx in rules:
        m = rx.search(text)
        if m:
            hits.append(m.group(0).strip())
    return hits


def _original_span(phrase: str, text: str) -> str:
    """Find the squashed phrase in the user's own spelling (e.g. 'kil him' -> 'kill himm')."""
    parts = []
    for word in phrase.split():
        parts.append("".join(re.escape(ch) + "+" for ch in word))
    rx = re.compile(r"[^a-z]+".join(parts), re.IGNORECASE)
    m = rx.search(re.sub(r"n['’]t\b", " not", str(text)))
    return m.group(0) if m else phrase


def check(text: str) -> dict:
    """Return {'self_harm': [...phrases], 'harm_to_others': [...phrases]} (empty lists if none)."""
    s = squash(text)
    return {"self_harm": [_original_span(p, text) for p in _find(_SELF, s)],
            "harm_to_others": [_original_span(p, text) for p in _find(_OTHERS, s)]}
