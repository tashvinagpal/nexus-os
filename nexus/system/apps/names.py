"""Names, aliases and fuzzy matching for apps and games.

"GTA", "gta 5", "gtav", "Grand Theft Auto" and "fire up grand theft auto V"
should all find "Grand Theft Auto V Legacy"; "chrome" should find "Google
Chrome". Not only exact strings, but high precision: ambiguity is reported,
never guessed.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable

_ROMAN_TO_DIGIT = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8",
                   "ix": "9", "x": "10"}
_DIGIT_TO_ROMAN = {d: r for r, d in _ROMAN_TO_DIGIT.items()}
# Edition/packaging words that don't identify a different program.
_EDITION = {"legacy", "enhanced", "remastered", "definitive", "edition", "goty", "deluxe", "ultimate",
            "complete", "premium", "standard", "x64", "x86", "64", "32", "bit", "64bit", "32bit", "new"}
_PUBLISHERS = {"google", "microsoft", "mozilla", "adobe", "valve", "epic", "rockstar", "riot", "nvidia", "amd",
               "intel", "oracle", "apple", "jetbrains", "autodesk", "corel"}
# Words people add around a name: "open THE chrome APP please", "GTA wala game".
_FILLER = {"the", "app", "application", "game", "please", "my", "wala", "wali", "vala", "program", "software",
           "nexus", "bhai", "yaar", "zara", "jaldi", "ek", "baar", "mera", "meri", "mere", "for", "me", "now",
           "abhi", "na", "plz", "pls", "a", "an", "ko", "to", "up"}
# Short generic exe names that must never become aliases.
_GENERIC_STEMS = {"app", "launcher", "setup", "update", "main", "game", "client", "start", "run", "bin", "player",
                  "helper", "service", "host", "server", "shell", "program", "application", "win64", "win32"}
CURATED_ALIASES = {
    "visual studio code": {"vs code", "vscode", "code"},
    "microsoft edge": {"edge"},
    "google chrome": {"chrome"},
    "mozilla firefox": {"firefox"},
    "obs studio": {"obs"},
    "file explorer": {"explorer", "files"},
    "epic games launcher": {"epic", "epic games", "epic launcher"},
    "rockstar games launcher": {"rockstar", "rockstar launcher"},
    "zoom workplace": {"zoom"},
    "valorant": {"valo"},
    "microsoft store": {"store"},
    "counter strike 2": {"cs2", "cs 2", "counter strike"},
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("&", " and ").replace("'", "")
    text = "".join(c if unicodedata.category(c)[0] in "LN" else " " for c in text)
    return " ".join(text.split())


def _numeral_variants(tokens: list[str]) -> set[str]:
    swapped = [(_ROMAN_TO_DIGIT.get(t) or _DIGIT_TO_ROMAN.get(t) or t) for t in tokens]
    return {" ".join(tokens), " ".join(swapped)}


def _is_numeral(token: str) -> bool:
    return token.isdigit() or token in _ROMAN_TO_DIGIT


def aliases_for(name: str, *, exe_stem: str | None = None, extra: Iterable[str] = ()) -> frozenset[str]:
    """All the ways someone might say this app's name."""
    out: set[str] = set()
    tokens = normalize(name).split()
    if not tokens:
        return frozenset()
    core = [t for t in tokens if t not in _EDITION] or tokens
    for variant in (tokens, core):
        out |= _numeral_variants(variant)
        if len(variant) > 1 and variant[0] in _PUBLISHERS:
            out |= _numeral_variants(variant[1:])
    # Acronyms for long names: "grand theft auto v" -> gta, gta v, gta 5, gtav, gta5
    words = [t for t in core if not _is_numeral(t) and t.isalpha()]
    numerals = [t for t in core if _is_numeral(t)]
    if len(words) >= 3:
        acronym = "".join(w[0] for w in words)
        out.add(acronym)
        for n in numerals[:1]:
            for num in {n, _ROMAN_TO_DIGIT.get(n, n), _DIGIT_TO_ROMAN.get(n, n)}:
                out |= {f"{acronym} {num}", f"{acronym}{num}"}
    if exe_stem:
        stem = normalize(exe_stem.replace("_", " "))
        if len(stem) >= 3 and stem not in _GENERIC_STEMS:
            out.add(stem)
            out.add(stem.replace(" ", ""))
    for base in list(out):
        out |= CURATED_ALIASES.get(base, set())
    out |= {normalize(e) for e in extra}
    return frozenset(a for a in out if len(a) >= 2 and a not in _FILLER)


_LEADING_VERB = re.compile(r"^\s*(?:please\s+)?(?:open|start|launch|run|play|fire\s+up|boot\s+up|load\s+up|"
                           r"bring\s+up|switch\s+to|khol(?:o|do)?|chala(?:o|do)?)\s+", re.I)
# Hinglish puts the verb last, often as two words: "GTA chala de", "Chrome khol do", "Steam start kar".
_HI_ENDING = r"(?:\s*(?:o|do|de|di|na|iye|dijiye|dena|doge))?"
TRAILING_VERBS = (rf"(?:khol{_HI_ENDING}|chala{_HI_ENDING}|chalu\s+kar{_HI_ENDING}|"
                  rf"(?:start|open|launch|shuru)\s+kar{_HI_ENDING}|laga{_HI_ENDING}|"
                  r"khel(?:te|na|o|enge)?(?:\s+ha(?:i|in))?)")
_TRAILING_VERB = re.compile(rf"\s+{TRAILING_VERBS}(?:\s+(?:na|please|zara|bhai|yaar|jaldi|re|ji))*\s*$", re.I)


def strip_launch_words(text: str) -> str:
    """'fire up Chrome' / 'GTA chala de' -> the name alone."""
    return _TRAILING_VERB.sub("", _LEADING_VERB.sub("", text.strip().rstrip("?!. "))).strip()


def query_variants(query: str) -> list[str]:
    tokens = [t for t in normalize(strip_launch_words(query)).split() if t not in _FILLER]
    if not tokens:
        return []
    return sorted(_numeral_variants(tokens))


def _token_match(q: str, a: str) -> bool:
    return q == a or (len(q) >= 3 and a.startswith(q))


def score(query_forms: list[str], aliases: frozenset[str]) -> float:
    """1.0 exact alias; ~0.9 every query word found in an alias; fuzzy for typos; 0 no match."""
    best = 0.0
    for q in query_forms:
        if q in aliases:
            return 1.0
        q_tokens = q.split()
        for alias in aliases:
            a_tokens = alias.split()
            if all(any(_token_match(t, a) for a in a_tokens) for t in q_tokens):
                best = max(best, 0.9 - 0.03 * max(0, len(a_tokens) - len(q_tokens)))
            elif len(q) >= 4 and abs(len(q) - len(alias)) <= 3:
                ratio = SequenceMatcher(None, q, alias).ratio()
                if ratio >= 0.8:
                    best = max(best, ratio * 0.9)
    return best
